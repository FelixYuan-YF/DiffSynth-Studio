import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
from ..core.gradient import gradient_checkpoint_forward

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def set_to_torch_norm(models):
    for model in models:
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.use_torch_norm = True


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_torch_norm = False
        self.normalized_shape = (dim,)

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        if self.use_torch_norm:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        else:        
            return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int] = (-1, -1), qk_norm: bool = True, eps: float = 1e-6, pos_encoder: str = "plucker", prope_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.pos_encoder = pos_encoder

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        if qk_norm:
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.attn = AttentionModule(self.num_heads)
        if pos_encoder == "prope":
            from .wan_video_camera_controller import prope_qkv
            self.prope_qkv_fn = prope_qkv
            self.o_prope = nn.Linear(dim, dim)
            if not self.o_prope.weight.is_meta:
                nn.init.zeros_(self.o_prope.weight)
            if self.o_prope.bias is not None and not self.o_prope.bias.is_meta:
                nn.init.zeros_(self.o_prope.bias)

    def forward(self, x, freqs, prope_viewmats=None, prope_Ks=None, video_shape=None):
        """Forward pass.

        Args:
            x: token sequence, shape (batch, seqlen, dim).
            freqs: RoPE frequencies (used only for the standard path).
            prope_viewmats: camera-to-world / world-to-camera matrices, shape
                (batch, cameras, 4, 4).  Required when ``pos_encoder="prope"``.
            prope_Ks: intrinsic matrices, shape (batch, cameras, 3, 3).  Optional
                even for PRoPE (falls back to the GTA formula when None).
            video_shape: (frames, height_patches, width_patches).  When
                provided the PRoPE patch grid is lazily reconfigured to match
                the current video resolution.
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if self.pos_encoder == "prope" and prope_viewmats is not None:
            # Dual-pathway logic: Original RoPE + PRoPE
            # Merge both paths into a single batched attention call for better
            # GPU utilisation (one kernel launch instead of two).
            w_patches = 40
            h_patches = 22
            if video_shape is not None:
                frames, h_patches, w_patches = video_shape

            orig_dtype = q.dtype

            # --- RoPE path: apply rotary embeddings to q/k ---
            q_rope = rope_apply(q, freqs, self.num_heads)  # (b, s, n*d)
            k_rope = rope_apply(k, freqs, self.num_heads)
            v_rope = v  # value is shared (no positional encoding)

            # --- PRoPE path: apply camera projection matrices in float32 ---
            q_bnsd = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads).float()
            k_bnsd = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads).float()
            v_bnsd = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads).float()

            q_prope_bnsd, k_prope_bnsd, v_prope_bnsd, apply_fn_o = self.prope_qkv_fn(
                q_bnsd, k_bnsd, v_bnsd,
                viewmats=prope_viewmats.float(),
                Ks=prope_Ks.float(),
                patches_x=w_patches,
                patches_y=h_patches,
            )

            # Cast PRoPE Q/K/V back to original dtype and layout (b, s, n*d)
            q_prope = rearrange(q_prope_bnsd, "b n s d -> b s (n d)").to(orig_dtype)
            k_prope = rearrange(k_prope_bnsd, "b n s d -> b s (n d)").to(orig_dtype)
            v_prope = rearrange(v_prope_bnsd, "b n s d -> b s (n d)").to(orig_dtype)

            # --- Batched attention: concat along batch dim, one kernel call ---
            q_all = torch.cat([q_rope, q_prope], dim=0)  # (2b, s, n*d)
            k_all = torch.cat([k_rope, k_prope], dim=0)
            v_all = torch.cat([v_rope, v_prope], dim=0)

            out_all = self.attn(q_all, k_all, v_all)      # (2b, s, n*d)

            out_rope, out_prope = out_all.chunk(2, dim=0)  # each (b, s, n*d)

            # Post-process PRoPE output: apply_fn_o (P matrix) in float32
            out_prope_bnsd = rearrange(out_prope, "b s (n d) -> b n s d", n=self.num_heads).float()
            out_prope_bnsd = apply_fn_o(out_prope_bnsd)
            out_prope = rearrange(out_prope_bnsd, "b n s d -> b s (n d)").to(orig_dtype)

            # Project and sum both paths
            x = self.o(out_rope) + self.o_prope(out_prope)
            return x
        else:
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            x = self.attn(q, k, v)
            return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, window_size: Tuple[int, int] = (-1, -1), eps: float = 1e-6,
                 pos_encoder: str = "plucker", prope_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, window_size, eps=eps, pos_encoder=pos_encoder, prope_config=prope_config)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, prope_viewmats=None, prope_Ks=None, video_shape=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, prope_viewmats=prope_viewmats, prope_Ks=prope_Ks, video_shape=video_shape))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: Optional[int] = None,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        # Positional encoding mode
        pos_encoder: str = "plucker", # "plucker" or "prope"
        prope_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.pos_encoder = pos_encoder

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps,
                     pos_encoder=pos_encoder, prope_config=prope_config)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        # We build control_adapter even if pos_encoder=="prope" to satisfy checkpoint 
        # weight loading, but bypass its forward pass in patchify.
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        bypass_adapter = self.pos_encoder == "prope"
        if self.control_adapter is not None and control_camera_latents_input is not None and not bypass_adapter:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # PRoPE branch introduces missing keys (o_prope weights) in the pre-trained state dictionary.
        # We allow missing keys here by enforcing strict=False, but manually verify that only o_prope is missing.
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            unexpected_missing = [k for k in missing_keys if "o_prope" not in k]
            if len(unexpected_missing) > 0 or len(unexpected_keys) > 0:
                error_msg = "Error(s) in loading state_dict for WanModel:\n"
                if len(unexpected_missing) > 0:
                    error_msg += f"Missing key(s) in state_dict: {unexpected_missing}\n"
                if len(unexpected_keys) > 0:
                    error_msg += f"Unexpected key(s) in state_dict: {unexpected_keys}\n"
                raise RuntimeError(error_msg)
        
        # Initialize o_prope meta-tensors so that `to(device)` does not fail.
        try:
            for module in self.modules():
                if hasattr(module, 'o_prope'):
                    o_prope = module.o_prope
                    if o_prope.weight.is_meta:
                        o_prope.weight = torch.nn.Parameter(torch.zeros_like(o_prope.weight, device='cpu'))
                    if o_prope.bias is not None and o_prope.bias.is_meta:
                        o_prope.bias = torch.nn.Parameter(torch.zeros_like(o_prope.bias, device='cpu'))
        except Exception:
            pass

        return missing_keys, unexpected_keys

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                viewmats: Optional[torch.Tensor] = None,
                Ks: Optional[torch.Tensor] = None,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x = self.patchify(x, control_camera_latents_input=kwargs.get("control_camera_latents_input", None))
        _, _, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # Pass video shape to PRoPE so it can lazily reconfigure the patch grid.
        video_shape = (f, h, w) if self.pos_encoder == "prope" else None
        for block in self.blocks:
            if self.training:
                x = gradient_checkpoint_forward(
                    block,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x, context, t_mod, freqs,
                    prope_viewmats=viewmats, prope_Ks=Ks, video_shape=video_shape
                )
            else:
                x = block(x, context, t_mod, freqs, prope_viewmats=viewmats, prope_Ks=Ks, video_shape=video_shape)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
