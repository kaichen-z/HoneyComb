from __future__ import annotations
import numpy as np
import torch
from adapter.corpus_utils import cell_rays, indexed_project, normalize_origin
from honeycomb import (
    AdaptiveBounds,
    query_projected_adaptive_readout,
    incoming_xyz_bounds_from_steps,
    incoming_xyz_inside_bounds,
    make_reserved_xyz_fixedt_bounds,
)
from shared_writer.field import write_field
from recurrent_writer.frame import IncrementalFrame
from recurrent_writer.incremental import StateField
from recurrent_writer.time_map import make_time_map


class _Placeholder:
    def __init__(self, bounds):
        self.bounds = bounds


class HexMemory:
    def __init__(
        self,
        latent_point_cloud_cls,
        *,
        num_time_steps,
        writer,
        device="cpu",
        output_device=None,
    ):
        if int(num_time_steps) < 1:
            raise ValueError(f"num_time_steps must be >= 1, got {num_time_steps}")
        if writer is None:
            raise ValueError(
                "HexMemory needs a writer (PlaneWriter or IncrementalWriter)"
            )
        self._lpc_cls = latent_point_cloud_cls
        incremental = hasattr(writer, "update")
        self._inc = writer if incremental else None
        self._writer = None if incremental else writer
        self._inc_state = None
        self.num_time_steps = int(num_time_steps)
        self._output_device = (
            None if output_device is None else torch.device(output_device)
        )
        self._device = torch.device(device)
        self._lpc = None
        self.model = None
        self.latent_mean = None
        self.latent_std = None
        self._tau = None
        self._write_dirs = None
        self._write_centers = None
        self._written: list[int] = []
        self.events: list[dict] = []
        self.read_log: list[dict] = []

    @property
    def points_world(self):
        self._require_initialized()
        return self._lpc.points_world

    @property
    def valid_mask(self):
        self._require_initialized()
        return self._lpc.valid_mask

    @property
    def latent_hw(self):
        self._require_initialized()
        return self._lpc.latent_hw

    @property
    def bounds(self):
        self._require_initialized()
        return self.model.bounds

    @property
    def written_times(self):
        return sorted(self._written)

    def initialize(self, *, depth, intrinsics, cam2world, latent, mask=None, time=0):
        if self._lpc is not None:
            raise RuntimeError("initialize() called twice; the field is persistent")
        t0 = self._check_time(time)
        lpc = self._lpc_cls.from_geometry(
            depth=depth,
            intrinsics=intrinsics,
            cam2world=cam2world,
            latent=latent,
            mask=mask,
            device="cpu",
        )
        if int(lpc.valid_mask.sum()) == 0:
            raise ValueError("no valid points in the seed frame")
        lat_h, lat_w = lpc.latent_hw
        n = lpc.points_world.shape[0]
        feats = lpc.features[lpc.valid_mask].float()
        self.latent_mean = feats.mean(0)
        self.latent_std = feats.std(0).clamp_min(1e-06)
        dirs_all, center = cell_rays(cam2world, lpc.intrinsics_latent, lat_h, lat_w)
        self._lpc = lpc
        self._tau = torch.full((n,), float(t0))
        self._write_dirs = dirs_all
        self._write_centers = center.reshape(1, 3).repeat(n, 1)
        bounds = AdaptiveBounds.from_steps(
            [t0],
            {t0: lpc.points_world},
            {t0: lpc.valid_mask},
            margin_frac=0.02,
            fixed_time_bounds=(0.0, float(self.num_time_steps - 1)),
        )
        self.events.append({"init": True, "time": t0, "bounds": bounds.to_json()})
        if self._inc is not None:
            v = lpc.valid_mask.bool()
            nv = int(v.sum())
            chunk = {
                "points_world": lpc.points_world[v].float(),
                "times": torch.full((nv,), float(t0)),
                "feats": (feats - self.latent_mean) / self.latent_std,
                "viewdirs": dirs_all[v].float(),
                "origins_world": center.reshape(1, 3)
                .expand(nv, 3)
                .float()
                .contiguous(),
            }
            self._inc_write(chunk, init_bounds=bounds)
            self._lpc.features = None
            self._write_dirs = None
            self._write_centers = None
        else:
            self._regenerate_field(init_bounds=bounds)
        self._written = [t0]

    def project(self, cam2world, intrinsics=None):
        pose = torch.as_tensor(
            np.asarray(cam2world) if not torch.is_tensor(cam2world) else cam2world,
            dtype=torch.float32,
        )
        if pose.shape != (4, 4):
            raise ValueError(
                f"project() takes ONE (4, 4) camera like LatentPointCloud.project, got {tuple(pose.shape)}; use project_many for a batch"
            )
        if intrinsics is None:
            self._require_initialized()
            intrinsics = self._lpc.intrinsics_latent
        latents, masks = self.project_many(pose, intrinsics)
        return (latents[0], masks[0])

    def project_many(self, cam2worlds, intrinsics):
        self._require_initialized()
        c2ws = self._as_pose_batch(cam2worlds)
        Ks = self._as_intrinsics_batch(intrinsics, c2ws.shape[0])
        lat_h, lat_w = self._lpc.latent_hw
        n_cells = lat_h * lat_w
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            sels, hits = ([], [])
            for c2w, K in zip(c2ws, Ks):
                sel, hit = indexed_project(self._lpc, c2w, K)
                sels.append(sel)
                hits.append(hit)
        finally:
            torch.set_num_threads(prev_threads)
        dirs, origins = ([], [])
        for c2w, K in zip(c2ws, Ks):
            d, center = cell_rays(c2w, K, lat_h, lat_w)
            dirs.append(d)
            origins.append(
                normalize_origin(center, self.model.bounds).expand(n_cells, 3)
            )
        out = query_projected_adaptive_readout(
            self.model,
            self._lpc.points_world,
            torch.stack(sels),
            torch.stack(hits),
            list(range(len(c2ws))),
            torch.stack(dirs),
            torch.stack(origins).contiguous(),
            self.latent_mean,
            self.latent_std,
            self._device,
            lat_h=lat_h,
            lat_w=lat_w,
            source_steps=self._tau,
        )
        self.read_log.append(
            {
                "n_poses": int(c2ws.shape[0]),
                "written_times_at_read": self.written_times,
                "n_points": int(self._lpc.points_world.shape[0]),
            }
        )
        latents = out["memory_latents_raw"][0].permute(1, 0, 2, 3).contiguous()
        masks = out["memory_mask"][0, 0].bool()
        if self._output_device is not None:
            latents = latents.to(self._output_device)
            masks = masks.to(self._output_device)
        return (latents, masks)

    def update(self, *, depths, intrinsics, cam2worlds, latents, times, masks=None):
        self._require_initialized()
        latents = torch.as_tensor(latents)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if latents.ndim != 4:
            raise ValueError(
                f"latents must be (C, h, w) or (T, C, h, w), got {tuple(latents.shape)}"
            )
        num_frames = latents.shape[0]
        times = [self._check_time(t) for t in times]
        if len(times) != num_frames:
            raise ValueError(f"{num_frames} latent frames but {len(times)} times")
        if len(set(times)) != len(times):
            raise ValueError(f"duplicate entries in times: {times}")
        dup = set(times) & set(self._written)
        if dup:
            raise ValueError(f"times already written to the field: {sorted(dup)}")
        depths = self._as_frame_batch(depths, num_frames, "depths", ndim=2)
        intr = self._as_frame_batch(intrinsics, num_frames, "intrinsics", ndim=2)
        c2ws = self._as_frame_batch(cam2worlds, num_frames, "cam2worlds", ndim=2)
        if masks is not None:
            masks = self._as_frame_batch(masks, num_frames, "masks", ndim=2)
        lat_h, lat_w = self._lpc.latent_hw
        new_pts, new_ok, appended = ({}, {}, [])
        for i in range(num_frames):
            frame_lpc = self._lpc_cls.from_geometry(
                depth=depths[i],
                intrinsics=intr[i],
                cam2world=c2ws[i],
                latent=latents[i].unsqueeze(1),
                mask=None if masks is None else masks[i],
                device="cpu",
            )
            valid = frame_lpc.valid_mask.bool()
            new_pts[times[i]] = frame_lpc.points_world
            new_ok[times[i]] = valid
            if valid.any():
                dirs_all, center = cell_rays(
                    c2ws[i], frame_lpc.intrinsics_latent, lat_h, lat_w
                )
                appended.append(
                    (
                        times[i],
                        frame_lpc.points_world[valid],
                        frame_lpc.features[valid],
                        dirs_all[valid],
                        center,
                    )
                )
        if not appended:
            return
        for t, pts, feats, dirs, center in appended:
            self._lpc.points_world = torch.cat([self._lpc.points_world, pts])
            if self._inc is None:
                self._lpc.features = torch.cat([self._lpc.features, feats])
            self._lpc.valid_mask = torch.cat(
                [
                    self._lpc.valid_mask.bool(),
                    torch.ones(pts.shape[0], dtype=torch.bool),
                ]
            )
            self._tau = torch.cat([self._tau, torch.full((pts.shape[0],), float(t))])
            if self._inc is None:
                self._write_dirs = torch.cat([self._write_dirs, dirs])
                self._write_centers = torch.cat(
                    [self._write_centers, center.reshape(1, 3).repeat(pts.shape[0], 1)]
                )
        written_times = sorted((t for t, *_ in appended))
        if self._inc is not None:
            old_bounds = self._inc_state.frame.bounds
            self._inc_write(self._chunk_from_appended(appended))
            nb = self._inc_state.frame.bounds
            self.events.append(
                {
                    "times": written_times,
                    "backend": "incremental",
                    "old_bounds": old_bounds.to_json(),
                    "new_bounds": nb.to_json(),
                    "warped": bool(
                        torch.any(torch.abs(old_bounds.lo - nb.lo) > 1e-08)
                        or torch.any(torch.abs(old_bounds.hi - nb.hi) > 1e-08)
                    ),
                }
            )
        else:
            event = _update_replacement_bounds(
                self.model, written_times, new_pts, new_ok, self.num_time_steps
            )
            self.events.append({"times": written_times, **event})
            self._regenerate_field()
        self._written = sorted(set(self._written) | set(written_times))

    def visible_points(self, cam2world, intrinsics):
        self._require_initialized()
        points_world = self._lpc.points_world[self._lpc.valid_mask.bool()]
        c2w = torch.as_tensor(cam2world, dtype=torch.float32)
        K = torch.as_tensor(intrinsics, dtype=torch.float32)
        world2cam = torch.inverse(c2w)
        points_cam = points_world @ world2cam[:3, :3].T + world2cam[:3, 3]
        z = points_cam[:, 2]
        u = points_cam[:, 0] * K[0, 0] / z + K[0, 2]
        v = points_cam[:, 1] * K[1, 1] / z + K[1, 2]
        height, width = self._lpc.latent_hw
        valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        return points_world[valid].detach().cpu().numpy().astype(np.float32)

    def _chunk_from_appended(self, appended):
        pts = torch.cat([p for _, p, _, _, _ in appended]).float()
        feats = torch.cat([f for _, _, f, _, _ in appended]).float()
        vd = torch.cat([d for _, _, _, d, _ in appended]).float()
        org = torch.cat(
            [c.reshape(1, 3).expand(p.shape[0], 3) for _, p, _, _, c in appended]
        ).float()
        times = torch.cat(
            [torch.full((p.shape[0],), float(t)) for t, p, _, _, _ in appended]
        )
        return {
            "points_world": pts,
            "times": times,
            "feats": (feats - self.latent_mean) / self.latent_std,
            "viewdirs": vd,
            "origins_world": org.contiguous(),
        }

    def _inc_write(self, chunk, init_bounds=None):
        dev = self._device
        c = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in chunk.items()}
        with torch.no_grad():
            if self._inc_state is None:
                if init_bounds is None:
                    raise RuntimeError("first incremental write needs init_bounds")
                meta = getattr(self._inc, "ckpt_meta", None) or {}
                tm = make_time_map(
                    meta.get("time_map") or "affine",
                    chunk_span=float(meta.get("chunk_span") or 8.0),
                )
                lo, hi = (init_bounds.lo.clone(), init_bounds.hi.clone())
                lo[3], hi[3] = (0.0, float(tm.chunk_span))
                frame = IncrementalFrame(AdaptiveBounds(lo=lo, hi=hi), tm)
                self._inc_state = self._inc.update(None, c, frame=frame)
                self.events.append(
                    {
                        "init_incremental": True,
                        "frame_bounds": self._inc_state.frame.bounds.to_json(),
                        "time_map": tm.name,
                        "chunk_span": float(tm.chunk_span),
                        "bounds_policy": self._inc.bounds_policy,
                        "fusion": self._inc.fusion,
                    }
                )
            else:
                self._inc_state = self._inc.update(self._inc_state, c)
        self.model = StateField(self._inc_state, self._inc.writer.reader)

    def _regenerate_field(self, *, init_bounds=None):
        bounds = init_bounds if self.model is None else self.model.bounds
        if self.model is None:
            self.model = _Placeholder(bounds)
        self.model = write_field(
            self._writer, self._write_inputs(), bounds, self.num_time_steps, self._device
        )

    def _write_inputs(self):
        valid = self._lpc.valid_mask.bool()
        lo = self.model.bounds.lo[:3]
        hi = self.model.bounds.hi[:3]
        org = 2.0 * (self._write_centers[valid] - lo) / (hi - lo).clamp_min(1e-08) - 1.0
        return (
            self._lpc.points_world[valid].float(),
            self._write_dirs[valid],
            org.contiguous(),
            (self._lpc.features[valid].float() - self.latent_mean) / self.latent_std,
            self._tau[valid],
        )

    def _check_time(self, time):
        t = int(time)
        if not 0 <= t < self.num_time_steps:
            raise ValueError(
                f"time {t} outside the padded rollout range [0, {self.num_time_steps - 1}]"
            )
        return t

    def _require_initialized(self):
        if self._lpc is None:
            raise RuntimeError("HexMemory used before initialize(); the field is empty")

    @staticmethod
    def _as_pose_batch(cam2worlds):
        poses = torch.as_tensor(
            np.asarray(cam2worlds) if not torch.is_tensor(cam2worlds) else cam2worlds,
            dtype=torch.float32,
        )
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"cam2worlds must be (4, 4) or (n, 4, 4), got {tuple(poses.shape)}"
            )
        return poses

    @staticmethod
    def _as_intrinsics_batch(intrinsics, n):
        K = torch.as_tensor(
            np.asarray(intrinsics) if not torch.is_tensor(intrinsics) else intrinsics,
            dtype=torch.float32,
        )
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(n, 3, 3)
        if K.ndim != 3 or K.shape[-2:] != (3, 3) or K.shape[0] != n:
            raise ValueError(
                f"intrinsics must be (3, 3) or ({n}, 3, 3), got {tuple(K.shape)}"
            )
        return K

    @staticmethod
    def _as_frame_batch(value, n, name, ndim):
        arr = torch.as_tensor(
            np.asarray(value) if not torch.is_tensor(value) else value,
            dtype=torch.float32,
        )
        if arr.ndim == ndim:
            arr = arr.unsqueeze(0)
        if arr.shape[0] != n:
            raise ValueError(f"{name} has {arr.shape[0]} frames, expected {n}")
        return arr


def make_memory_factory(latent_point_cloud_cls, *, writer):
    def factory(*, depth, intrinsics, cam2world, latent, mask, device, num_time_steps):
        memory = HexMemory(
            latent_point_cloud_cls,
            num_time_steps=num_time_steps,
            writer=writer.to(device),
            device=device,
            output_device=device,
        )
        memory.initialize(
            depth=depth,
            intrinsics=intrinsics,
            cam2world=cam2world,
            latent=latent,
            mask=mask,
        )
        return memory

    return factory


def _update_replacement_bounds(
    model,
    steps,
    pts,
    ok,
    K,
    margin_frac=0.02,
    growth_factor=1.5,
    needed_factor=1.1,
    eps=1e-06,
):
    inc_lo, inc_hi = incoming_xyz_bounds_from_steps(
        steps, pts, ok, margin_frac=margin_frac
    )
    event_base = {
        "incoming_xyz_lo": [float(x) for x in inc_lo.tolist()],
        "incoming_xyz_hi": [float(x) for x in inc_hi.tolist()],
        "reserve_growth_factor": float(growth_factor),
        "reserve_needed_factor": float(needed_factor),
        "reserve_eps": float(eps),
    }
    if incoming_xyz_inside_bounds(model.bounds, inc_lo, inc_hi, eps=eps):
        return {
            **event_base,
            "old_bounds": model.bounds.to_json(),
            "new_bounds": model.bounds.to_json(),
            "warped": False,
            "reason": "inside_reserved_bounds",
        }
    new_bounds = make_reserved_xyz_fixedt_bounds(
        model.bounds,
        inc_lo,
        inc_hi,
        K,
        growth_factor=growth_factor,
        needed_factor=needed_factor,
        eps=eps,
    )
    event = model.warp_to_bounds(new_bounds, min_growth_frac=0.0, force=True)
    event.update(event_base)
    event["reason"] = "exited_reserved_bounds"
    if not event["warped"]:
        raise RuntimeError("reserved bounds changed without warping")
    return event
