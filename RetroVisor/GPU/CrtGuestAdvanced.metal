// -----------------------------------------------------------------------------
// CRT Guest Advanced – Metal compute‑kernel port
//
// Original GLSL shader: Copyright (C) 2018-2025 guest(r) and Dr. Venom
// Licensed under the GNU General Public License v2+
//
// MSL port for RetroVisor.
// The original 12‑pass Vulkan/GLSL pipeline is collapsed into 10 compute
// kernels that are dispatched sequentially from Swift.
// -----------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared uniforms – one struct passed to every kernel via setBytes
// ---------------------------------------------------------------------------

namespace crtguest {

struct Uniforms {

    // -- Sizes --
    float2 sourceSize;     // original content size (pixels)
    float2 outputSize;     // output/window size (pixels)
    uint   frameCount;

    // -- Afterglow --
    float PR;              // persistence R   0.32
    float PG;              // persistence G   0.32
    float PB;              // persistence B   0.32
    float esrc;            // afterglow source 1.0
    float bth;             // afterglow threshold 4.0

    // -- Pre‑shaders / color --
    float AS;              // afterglow strength 0.20
    float agsat;           // afterglow saturation 0.50
    float CS;              // display gamut 0
    float CP;              // CRT profile 0
    float TNTC;            // LUT colors 0
    float LS;              // LUT size 32
    float WP;              // color temperature 0
    float wp_saturation;   // saturation 1.0
    float pre_bb;          // brightness 1.0
    float contr;           // contrast 0.0
    float pre_gc;          // gamma correct adj 1.0
    float sega_fix;        // sega brightness 0
    float BP;              // raise black level 0
    float vigstr;          // vignette strength 0
    float vigdef;          // vignette size 1.0

    // -- Raster bloom / avg lum --
    float lsmooth;         // 0.70
    float lsdev;           // 0.0
    float OS;              // overscan mode 1
    float BLOOM;           // raster bloom % 0

    // -- Gamma / interlace --
    float GAMMA_INPUT;     // 2.4
    float gamma_out;       // 2.4
    float inter;           // interlace trigger 375
    float interm;          // interlace mode 1
    float iscan;           // interlace scanline 0.20
    float intres;          // internal resolution Y 0
    float downsample_levelx; // 0
    float downsample_levely; // 0
    float iscans;          // interlace saturation 0.25
    float vga_mode;        // 0
    float hiscan;          // 0

    // -- Magic glow --
    float m_glow;          // 0
    float m_glow_cutoff;   // 0.12
    float m_glow_low;      // 0.35
    float m_glow_high;     // 5.0
    float m_glow_dist;     // 1.0
    float m_glow_mask;     // 1.0

    // -- Glow pass --
    float FINE_GLOW;       // 1.0
    float SIZEH;           // 6
    float SIGMA_H;         // 1.20
    float SIZEV;           // 6
    float SIGMA_V;         // 1.20

    // -- Bloom pass --
    float FINE_BLOOM;      // 1.0
    float SIZEHB;          // 3
    float SIGMA_HB;        // 0.75
    float SIZEVB;          // 3
    float SIGMA_VB;        // 0.60

    // -- Brightness --
    float glow;            // 0.08
    float bloom;           // 0
    float mask_bloom;      // 0
    float bloom_dist;      // 0
    float halation;        // 0
    float bmask1;          // 0
    float hmask1;          // 0.35
    float gamma_c;         // 1.0
    float gamma_c2;        // 1.0
    float brightboost;     // 1.40
    float brightboost1;    // 1.10
    float clips;           // 0

    // -- Scanline --
    float gsl;             // 0
    float scanline1;       // 6
    float scanline2;       // 8
    float beam_min;        // 1.30
    float beam_max;        // 1.00
    float tds;             // 0
    float beam_size;       // 0.60
    float scans;           // 0.50
    float scan_falloff;    // 1.0
    float spike;           // 1.0
    float scangamma;       // 2.40
    float rolling_scan;    // 0
    float no_scanlines;    // 0

    // -- Filtering --
    float h_sharp;         // 5.20
    float s_sharp;         // 0.50
    float ring;            // 0
    float smart_ei;        // 0
    float ei_limit;        // 0.25
    float sth;             // 0.23

    // -- Screen --
    float TATE;            // 0
    float IOS;             // 0
    float warpX;           // 0
    float warpY;           // 0
    float c_shape;         // 0.25
    float overscanX;       // 0
    float overscanY;       // 0
    float VShift;          // 0

    // -- Mask --
    float shadowMask;      // 0
    float maskstr;         // 0.3
    float mcut;            // 1.10
    float maskboost;       // 1.0
    float masksize;        // 1
    float mask_zoom;       // 0
    float mzoom_sh;        // 0
    float mshift;          // 0
    float mask_layout;     // 0
    float maskDark;        // 0.5
    float maskLight;       // 1.5
    float mask_gamma;      // 2.40
    float slotmask;        // 0
    float slotmask1;       // 0
    float slotwidth;       // 0
    float double_slot;     // 2
    float slotms;          // 1
    float smoothmask;      // 0
    float smask_mit;       // 0
    float bmask;           // 0
    float mclip;           // 0
    float pr_scan;         // 0.10
    float edgemask;        // 0

    // -- Border / corner --
    float csize;           // 0
    float bsize1;          // 0
    float sborder;         // 0.75

    // -- Hum bar --
    float barspeed;        // 50
    float barintensity;    // 0
    float bardir;          // 0

    // -- Deconvergence --
    float dctypex;         // 0
    float dctypey;         // 0
    float deconrr;         // 0
    float deconrg;         // 0
    float deconrb;         // 0
    float deconrry;        // 0
    float deconrgy;        // 0
    float deconrby;        // 0
    float decons;          // 1.0

    // -- Post / noise --
    float addnoised;       // 0
    float noiseresd;       // 2
    float noisetype;       // 0
    float post_br;         // 1.0
    float oimage;          // 0
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline float3 plant(float3 tar, float r)
{
    float t = max(max(tar.r, tar.g), tar.b) + 0.00001;
    return tar * r / t;
}

// ---------- colour-distance (avg-lum) ----------
static inline float cdist(float3 A, float3 B)
{
    float r = 0.5 * (A.r + B.r);
    float3 d = A - B;
    float3 c = float3(2.0 + r, 4.0, 3.0 - r);
    return sqrt(dot(c * d, d)) / 3.0;
}

// =========================================================================
// Pass 2 – Afterglow (feedback)
// =========================================================================

kernel void guestAfterglowPass(
    texture2d<float, access::read>   source      [[texture(0)]],
    texture2d<float, access::read>   feedback    [[texture(1)]],
    texture2d<float, access::write>  outTex      [[texture(2)]],
    constant Uniforms &u                         [[buffer(0)]],
    uint2 gid                                    [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    float2 invSrc = 1.0 / u.sourceSize;
    float2 dx = float2(invSrc.x, 0.0);
    float2 dy = float2(0.0, invSrc.y);
    float2 uv = (float2(gid) + 0.5) / u.sourceSize;

    // Read 5-tap diamond from source
    float3 color0 = source.read(gid).rgb;
    uint2 g1 = uint2(clamp(int2(gid) + int2(-1, 0), int2(0), int2(outTex.get_width()-1, outTex.get_height()-1)));
    uint2 g2 = uint2(clamp(int2(gid) + int2( 1, 0), int2(0), int2(outTex.get_width()-1, outTex.get_height()-1)));
    uint2 g3 = uint2(clamp(int2(gid) + int2( 0,-1), int2(0), int2(outTex.get_width()-1, outTex.get_height()-1)));
    uint2 g4 = uint2(clamp(int2(gid) + int2( 0, 1), int2(0), int2(outTex.get_width()-1, outTex.get_height()-1)));

    float3 color1 = source.read(g1).rgb;
    float3 color2 = source.read(g2).rgb;
    float3 color3 = source.read(g3).rgb;
    float3 color4 = source.read(g4).rgb;

    float3 color = (2.5 * color0 + color1 + color2 + color3 + color4) / 6.5;

    float3 accumulate = feedback.read(gid).rgb;

    float b = u.bth / 255.0;
    float c = max(max(color0.r, color0.g), color0.b);
    float w = smoothstep(b, 2.0 * b, c);

    float3 result = mix(max(mix(color, accumulate, 0.49 + float3(u.PR, u.PG, u.PB)) - 1.25 / 255.0, 0.0), color, w);

    outTex.write(float4(result, w), gid);
}

// =========================================================================
// Pass 3 – Pre-shaders (color correction, LUT, vignette)
// =========================================================================

// Colour profile matrices – constant arrays
constant float3x3 Profile0 = float3x3(
    float3(0.412391, 0.212639, 0.019331),
    float3(0.357584, 0.715169, 0.119195),
    float3(0.180481, 0.072192, 0.950532));
constant float3x3 Profile1 = float3x3(
    float3(0.430554, 0.222004, 0.020182),
    float3(0.341550, 0.706655, 0.129553),
    float3(0.178352, 0.071341, 0.939322));
constant float3x3 Profile2 = float3x3(
    float3(0.396686, 0.210299, 0.006131),
    float3(0.372504, 0.713766, 0.115356),
    float3(0.181266, 0.075936, 0.967571));
constant float3x3 Profile3 = float3x3(
    float3(0.393521, 0.212376, 0.018739),
    float3(0.365258, 0.701060, 0.111934),
    float3(0.191677, 0.086564, 0.958385));
constant float3x3 Profile4 = float3x3(
    float3(0.392258, 0.209410, 0.016061),
    float3(0.351135, 0.725680, 0.093636),
    float3(0.166603, 0.064910, 0.850324));
constant float3x3 Profile5 = float3x3(
    float3(0.377923, 0.195679, 0.010514),
    float3(0.317366, 0.722319, 0.097826),
    float3(0.207738, 0.082002, 1.076960));

constant float3x3 ToSRGB = float3x3(
    float3( 3.240970,-0.969244, 0.055630),
    float3(-1.537383, 1.875968,-0.203977),
    float3(-0.498611, 0.041555, 1.056972));
constant float3x3 ToModern = float3x3(
    float3( 2.791723,-0.894766, 0.041678),
    float3(-1.173165, 1.815586,-0.130886),
    float3(-0.440973, 0.032000, 1.002034));
constant float3x3 ToDCI = float3x3(
    float3( 2.973422,-0.867605, 0.045031),
    float3(-1.110433, 1.843757,-0.095697),
    float3(-0.480247, 0.024743, 1.201215));
constant float3x3 ToAdobe = float3x3(
    float3( 2.041588,-0.969244, 0.013444),
    float3(-0.565007, 1.875968,-0.11836),
    float3(-0.344731, 0.041555, 1.015175));
constant float3x3 ToREC = float3x3(
    float3( 1.716651,-0.666684, 0.017640),
    float3(-0.355671, 1.616481,-0.042771),
    float3(-0.253366, 0.015769, 0.942103));
constant float3x3 ToP3 = float3x3(
    float3( 2.493509,-0.829473, 0.0358512),
    float3(-0.931388, 1.762630,-0.0761839),
    float3(-0.402712, 0.023624, 0.9570296));

constant float3x3 D65_to_D55 = float3x3(
    float3(0.4850339153, 0.2500956126, 0.0227359648),
    float3(0.3488957224, 0.6977914447, 0.1162985741),
    float3(0.1302823568, 0.0521129427, 0.6861537456));
constant float3x3 D65_to_D93 = float3x3(
    float3(0.3412754080, 0.1759701322, 0.0159972847),
    float3(0.3646170520, 0.7292341040, 0.1215390173),
    float3(0.2369894093, 0.0947957637, 1.2481442225));

static inline float3 fix_lut(float3 lutcolor, float3 ref)
{
    float r = length(ref);
    float l = length(lutcolor);
    float m = max(max(ref.r, ref.g), ref.b);
    float3 n = normalize(lutcolor + 0.0000001) * mix(r, l, pow(m, 1.25));
    return mix(lutcolor, n, 1.0); // LUTBR = 1.0
}

static inline float vignette(float2 pos, float vigstr_, float vigdef_, float2 origSize)
{
    float2 b = float2(vigdef_, vigdef_) * float2(1.0, origSize.x / origSize.y) * 0.125;
    pos = clamp(pos, 0.0, 1.0);
    pos = abs(2.0 * (pos - 0.5));
    float2 res = mix(float2(0.0), float2(1.0), smoothstep(float2(1.0), float2(1.0) - b, sqrt(pos)));
    res = pow(res, float2(0.70));
    return max(mix(1.0, sqrt(res.x * res.y), vigstr_), 0.0);
}

static inline float contrastFn(float x, float contr_)
{
    return max(mix(x, smoothstep(0.0, 1.0, x), contr_), 0.0);
}

static inline float3 pgc(float3 c, float pre_gc_)
{
    float mc = max(max(c.r, c.g), c.b);
    float mg = pow(mc, 1.0 / pre_gc_);
    return c * mg / (mc + 1e-8);
}

kernel void guestPreShadersPass(
    texture2d<float, access::read>    stockPass    [[texture(0)]],
    texture2d<float, access::read>    afterglow    [[texture(1)]],
    texture2d<float, access::sample>  lut1         [[texture(2)]],
    texture2d<float, access::sample>  lut2         [[texture(3)]],
    texture2d<float, access::sample>  lut3         [[texture(4)]],
    texture2d<float, access::sample>  lut4         [[texture(5)]],
    texture2d<float, access::write>   outTex       [[texture(6)]],
    constant Uniforms &u                           [[buffer(0)]],
    sampler linearSampler                          [[sampler(0)]],
    uint2 gid                                      [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    float2 uv = (float2(gid) + 0.5) / u.sourceSize;

    float4 imgColor = float4(stockPass.read(gid).rgb, 1.0);
    float4 aftglow = afterglow.read(gid);

    float w = 1.0 - aftglow.w;

    float l = length(aftglow.rgb);
    aftglow.rgb = u.AS * w * normalize(pow(aftglow.rgb + 0.01, float3(u.agsat))) * l;
    float bp = w * u.BP / 255.0;

    if (u.sega_fix > 0.5) imgColor.rgb = imgColor.rgb * (255.0 / 239.0);
    imgColor.rgb = min(imgColor.rgb, 1.0);

    float3 color = imgColor.rgb;

    // LUT application
    int tntc = int(u.TNTC);
    if (tntc != 0)
    {
        float lutlow = 5.0 / 255.0;
        float invLS = 1.0 / u.LS;
        float3 lut_ref = imgColor.rgb + lutlow * (1.0 - pow(imgColor.rgb, float3(0.333)));
        float lutb = lut_ref.b * (1.0 - 0.5 * invLS);
        lut_ref.rg = lut_ref.rg * (1.0 - invLS) + 0.5 * invLS;
        float tile1 = ceil(lutb * (u.LS - 1.0));
        float tile0 = max(tile1 - 1.0, 0.0);
        float f = fract(lutb * (u.LS - 1.0));
        if (f == 0.0) f = 1.0;
        float2 coord0 = float2(tile0 + lut_ref.r, lut_ref.g) * float2(invLS, 1.0);
        float2 coord1 = float2(tile1 + lut_ref.r, lut_ref.g) * float2(invLS, 1.0);
        float4 color1_, color2_, res_;

        if (tntc == 1) {
            color1_ = lut1.sample(linearSampler, coord0);
            color2_ = lut1.sample(linearSampler, coord1);
        } else if (tntc == 2) {
            color1_ = lut2.sample(linearSampler, coord0);
            color2_ = lut2.sample(linearSampler, coord1);
        } else if (tntc == 3) {
            color1_ = lut3.sample(linearSampler, coord0);
            color2_ = lut3.sample(linearSampler, coord1);
        } else {
            color1_ = lut4.sample(linearSampler, coord0);
            color2_ = lut4.sample(linearSampler, coord1);
        }
        res_ = mix(color1_, color2_, f);
        res_.rgb = fix_lut(res_.rgb, imgColor.rgb);
        color = mix(imgColor.rgb, res_.rgb, min(float(u.TNTC), 1.0));
    }

    float3 c = clamp(color, 0.0, 1.0);

    float p;
    float3x3 m_out;

    if      (u.CS == 0.0) { p = 2.2; m_out = ToSRGB;   }
    else if (u.CS == 1.0) { p = 2.2; m_out = ToModern;  }
    else if (u.CS == 2.0) { p = 2.6; m_out = ToDCI;     }
    else if (u.CS == 3.0) { p = 2.2; m_out = ToAdobe;   }
    else if (u.CS == 4.0) { p = 2.4; m_out = ToREC;     }
    else                   { p = 2.2; m_out = ToP3;      }

    color = pow(c, float3(p));

    float3x3 m_in = Profile0;
    if      (u.CP == 0.0) m_in = Profile0;
    else if (u.CP == 1.0) m_in = Profile1;
    else if (u.CP == 2.0) m_in = Profile2;
    else if (u.CP == 3.0) m_in = Profile3;
    else if (u.CP == 4.0) m_in = Profile4;
    else if (u.CP == 5.0) m_in = Profile5;

    color = m_in * color;
    color = m_out * color;
    color = clamp(color, 0.0, 1.0);
    color = pow(color, float3(1.0 / p));

    if (u.CP == -1.0) color = c;

    // Saturation
    float3 scolor1 = plant(pow(color, float3(u.wp_saturation)), max(max(color.r, color.g), color.b));
    float luma = dot(color, float3(0.299, 0.587, 0.114));
    float3 scolor2 = mix(float3(luma), color, u.wp_saturation);
    color = (u.wp_saturation > 1.0) ? scolor1 : scolor2;

    color = plant(color, contrastFn(max(max(color.r, color.g), color.b), u.contr));

    p = 2.2;
    color = clamp(color, 0.0, 1.0);
    color = pow(color, float3(p));

    float3 warmer = D65_to_D55 * color;
    warmer = ToSRGB * warmer;
    float3 cooler = D65_to_D93 * color;
    cooler = ToSRGB * cooler;

    float m = abs(u.WP) / 100.0;
    float3 comp = (u.WP < 0.0) ? cooler : warmer;
    color = mix(color, comp, m);
    color = pow(max(color, 0.0), float3(1.0 / p));
    color = pgc(color, u.pre_gc);

    if (u.BP > -0.5) {
        color = color + aftglow.rgb + bp;
    } else {
        color = max(color + u.BP / 255.0, 0.0) /
                (1.0 + u.BP / 255.0 * step(-u.BP / 255.0, max(max(color.r, color.g), color.b))) + aftglow.rgb;
    }

    color = min(color * u.pre_bb, 1.0);

    outTex.write(float4(color, vignette(uv, u.vigstr, u.vigdef, u.sourceSize)), gid);
}

// =========================================================================
// Pass 4 – Average Luminance + edge coefficients (feedback)
// =========================================================================

kernel void guestAvgLumPass(
    texture2d<float, access::sample>  source   [[texture(0)]],
    texture2d<float, access::read>    feedback  [[texture(1)]],
    texture2d<float, access::write>   outTex    [[texture(2)]],
    constant Uniforms &u                        [[buffer(0)]],
    sampler samp                                [[sampler(0)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    float2 srcSize = float2(outTex.get_width(), outTex.get_height());
    float2 uv = (float2(gid) + 0.5) / srcSize;

    // Mipmap-based average luminance (sample at 4 corners with high LOD)
    float m = max(log2(srcSize.x), log2(srcSize.y));
    m = floor(max(m, 1.0)) - 1.0;

    float ltotal = 0.0;
    ltotal += length(source.sample(samp, float2(0.3, 0.3), level(m)).rgb);
    ltotal += length(source.sample(samp, float2(0.3, 0.7), level(m)).rgb);
    ltotal += length(source.sample(samp, float2(0.7, 0.3), level(m)).rgb);
    ltotal += length(source.sample(samp, float2(0.7, 0.7), level(m)).rgb);
    ltotal *= 0.25;
    ltotal = pow(0.577350269 * ltotal, 0.70);

    float lhistory = feedback.read(uint2(outTex.get_width()/2, outTex.get_height()/2)).a;
    ltotal = mix(ltotal, lhistory, min(mix(u.lsmooth, u.lsmooth + u.lsdev, ltotal), 0.99));

    // Edge detection coefficients
    float2 dx2 = float2(1.0 / srcSize.x, 0.0);
    float2 x2 = 2.0 * dx2;

    float3 l1 = source.sample(samp, uv).rgb;
    float3 r1 = source.sample(samp, uv + dx2).rgb;
    float3 l2 = source.sample(samp, uv - dx2).rgb;
    float3 r2 = source.sample(samp, uv + x2).rgb;

    float c1 = cdist(l2, l1);
    float c2 = cdist(l1, r1);
    float c3 = cdist(r2, r1);

    outTex.write(float4(c1, c2, c3, ltotal), gid);
}

// =========================================================================
// Pass 5 – Linearize + Interlace
// =========================================================================

kernel void guestLinearizePass(
    texture2d<float, access::sample>  prePass  [[texture(0)]],
    texture2d<float, access::write>   outTex   [[texture(1)]],
    constant Uniforms &u                       [[buffer(0)]],
    sampler samp                               [[sampler(0)]],
    uint2 gid                                  [[thread_position_in_grid]])
{
    if (gid.x >= outTex.get_width() || gid.y >= outTex.get_height()) return;

    float2 srcSize = float2(outTex.get_width(), outTex.get_height());
    float2 uv = (float2(gid) + 0.5) / srcSize;

    float3 c1, c2;

    // Downsampling fetch
    if ((u.downsample_levelx + u.downsample_levely) > 0.025) {
        float2 dx_ds = float2(1.0 / srcSize.x, 0.0) * u.downsample_levelx;
        float2 dy_ds = float2(0.0, 1.0 / srcSize.y) * u.downsample_levely;
        float2 d1 = dx_ds + dy_ds;
        float2 d2 = dx_ds - dy_ds;
        float sum = 15.0;
        c1 = 3.0 * prePass.sample(samp, uv).rgb +
             2.0 * prePass.sample(samp, uv + dx_ds).rgb +
             2.0 * prePass.sample(samp, uv - dx_ds).rgb +
             2.0 * prePass.sample(samp, uv + dy_ds).rgb +
             2.0 * prePass.sample(samp, uv - dy_ds).rgb +
             prePass.sample(samp, uv + d1).rgb +
             prePass.sample(samp, uv - d1).rgb +
             prePass.sample(samp, uv + d2).rgb +
             prePass.sample(samp, uv - d2).rgb;
        c1 /= sum;

        float2 uv2 = uv + float2(0.0, 1.0 / u.sourceSize.y);
        c2 = 3.0 * prePass.sample(samp, uv2).rgb +
             2.0 * prePass.sample(samp, uv2 + dx_ds).rgb +
             2.0 * prePass.sample(samp, uv2 - dx_ds).rgb +
             2.0 * prePass.sample(samp, uv2 + dy_ds).rgb +
             2.0 * prePass.sample(samp, uv2 - dy_ds).rgb +
             prePass.sample(samp, uv2 + d1).rgb +
             prePass.sample(samp, uv2 - d1).rgb +
             prePass.sample(samp, uv2 + d2).rgb +
             prePass.sample(samp, uv2 - d2).rgb;
        c2 /= sum;
    } else {
        c1 = prePass.sample(samp, uv).rgb;
        c2 = prePass.sample(samp, uv + float2(0.0, 1.0 / u.sourceSize.y)).rgb;
    }

    float3 c = c1;
    float intera = 1.0;
    float gamma_in = u.GAMMA_INPUT;

    float m1 = max(max(c1.r, c1.g), c1.b);
    float m2 = max(max(c2.r, c2.g), c2.b);
    float3 df = abs(c1 - c2);
    float d = max(max(df.r, df.g), df.b);
    if (u.interm == 2.0) d = mix(0.1 * d, 10.0 * d, step(m1 / (m2 + 0.0001), m2 / (m1 + 0.0001)));

    float yres_div = 1.0;
    if (u.intres > 1.25) yres_div = u.intres;

    bool hscan = (u.hiscan > 0.5);

    if ((u.inter <= u.sourceSize.y / yres_div && u.interm > 0.5 && u.intres != 1.0 && u.intres != 0.5 && u.vga_mode < 0.5) || hscan) {
        intera = 0.25;
        float line_no = floor(fmod(u.sourceSize.y * uv.y, 2.0));
        float frame_no = floor(fmod(float(u.frameCount), 2.0));
        float ii = abs(line_no - frame_no);

        if (u.interm < 3.5 || u.interm > 5.5) {
            if (u.interm == 6.0) {
                c = mix(c2, c1, ii);
            } else {
                c2 = plant(mix(c2, c2 * c2, u.iscans), max(max(c2.r, c2.g), c2.b));
                float r = max(m1 * ii, (1.0 - u.iscan) * min(m1, m2));
                c = plant(mix(mix(c1, c2, min(mix(m1, 1.0 - m2, min(m1, 1.0 - m1)) / (d + 0.00001), 1.0)), c1, ii), r);
                if (u.interm == 3.0) c = (1.0 - 0.5 * u.iscan) * mix(c2, c1, ii);
            }
        }
        if (u.interm == 4.0) {
            c = plant(mix(c, c * c, 0.5 * u.iscans), max(max(c.r, c.g), c.b)) * (1.0 - 0.5 * u.iscan);
        }
        if (u.interm == 5.0) {
            c = mix(c2, c1, 0.5);
            c = plant(mix(c, c * c, 0.5 * u.iscans), max(max(c.r, c.g), c.b)) * (1.0 - 0.5 * u.iscan);
        }
        if (hscan) c = c1;
    }

    if (u.vga_mode > 0.5) {
        c = c1;
        if (u.inter <= u.sourceSize.y) intera = 0.75; else intera = 0.5;
    }

    c = pow(c, float3(gamma_in));

    // Pack gamma info into alpha: left half = intera, right half = 1/gamma_in
    float alpha_out;
    if (uv.x > 0.5) alpha_out = intera; else alpha_out = 1.0 / gamma_in;

    outTex.write(float4(c, alpha_out), gid);
}

// =========================================================================
// Pass 6 – Gaussian Horizontal (glow)
// =========================================================================

kernel void guestGaussianHPass(
    texture2d<float, access::sample>  linearize [[texture(0)]],
    texture2d<float, access::write>   outTex    [[texture(1)]],
    constant Uniforms &u                        [[buffer(0)]],
    sampler samp                                [[sampler(0)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    uint outW = outTex.get_width();
    uint outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float FINE_G = (u.FINE_GLOW > 0.5) ? u.FINE_GLOW : mix(0.75, 0.5, -u.FINE_GLOW);
    float4 srcSize1 = float4(u.sourceSize.x, u.sourceSize.y, 1.0/u.sourceSize.x, 1.0/u.sourceSize.y)
                    * float4(FINE_G, FINE_G, 1.0/FINE_G, 1.0/FINE_G);

    float2 uv = (float2(gid) + 0.5) / float2(outW, outH);
    float f = fract(srcSize1.x * uv.x);
    f = 0.5 - f;
    float2 tex = floor(srcSize1.xy * uv) * srcSize1.zw + 0.5 * srcSize1.zw;
    float3 color = float3(0.0);
    float2 dx_ = float2(srcSize1.z, 0.0);

    float invsqrsigma = 1.0 / (2.0 * u.SIGMA_H * u.SIGMA_H);
    float wsum = 0.0;

    for (float n = -u.SIZEH; n <= u.SIZEH; n += 1.0) {
        float3 pixel = linearize.sample(samp, tex + n * dx_).rgb;
        if (u.m_glow > 0.5) {
            pixel = max(pixel - u.m_glow_cutoff, 0.0);
            pixel = plant(pixel, max(max(max(pixel.r, pixel.g), pixel.b) - u.m_glow_cutoff, 0.0));
        }
        float w_ = exp(-(n + f) * (n + f) * invsqrsigma);
        color += w_ * pixel;
        wsum += w_;
    }
    color /= wsum;
    outTex.write(float4(color, 1.0), gid);
}

// =========================================================================
// Pass 7 – Gaussian Vertical (glow) → GlowPass
// =========================================================================

kernel void guestGaussianVPass(
    texture2d<float, access::sample>  source   [[texture(0)]],
    texture2d<float, access::write>   outTex   [[texture(1)]],
    constant Uniforms &u                       [[buffer(0)]],
    sampler samp                               [[sampler(0)]],
    uint2 gid                                  [[thread_position_in_grid]])
{
    uint outW = outTex.get_width();
    uint outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float FINE_G = (u.FINE_GLOW > 0.5) ? u.FINE_GLOW : mix(0.75, 0.5, -u.FINE_GLOW);
    // Mix source width with original height, scaled by FINE_GLOW
    float4 srcSize1 = float4(float(outW), u.sourceSize.y, 1.0/float(outW), 1.0/u.sourceSize.y)
                    * float4(FINE_G, FINE_G, 1.0/FINE_G, 1.0/FINE_G);

    float2 uv = (float2(gid) + 0.5) / float2(outW, outH);
    float f = fract(srcSize1.y * uv.y);
    f = 0.5 - f;
    float2 tex = floor(srcSize1.xy * uv) * srcSize1.zw + 0.5 * srcSize1.zw;
    float3 color = float3(0.0);
    float2 dy_ = float2(0.0, srcSize1.w);

    float invsqrsigma = 1.0 / (2.0 * u.SIGMA_V * u.SIGMA_V);
    float wsum = 0.0;

    for (float n = -u.SIZEV; n <= u.SIZEV; n += 1.0) {
        float3 pixel = source.sample(samp, tex + n * dy_).rgb;
        float w_ = exp(-(n + f) * (n + f) * invsqrsigma);
        color += w_ * pixel;
        wsum += w_;
    }
    color /= wsum;
    outTex.write(float4(color, 1.0), gid);
}

// =========================================================================
// Pass 8 – Bloom Horizontal
// =========================================================================

kernel void guestBloomHPass(
    texture2d<float, access::sample>  linearize [[texture(0)]],
    texture2d<float, access::write>   outTex    [[texture(1)]],
    constant Uniforms &u                        [[buffer(0)]],
    sampler samp                                [[sampler(0)]],
    uint2 gid                                   [[thread_position_in_grid]])
{
    uint outW = outTex.get_width();
    uint outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float FINE_B = (u.FINE_BLOOM > 0.5) ? u.FINE_BLOOM : mix(0.75, 0.5, -u.FINE_BLOOM);
    float4 srcSize1 = float4(u.sourceSize.x, u.sourceSize.y, 1.0/u.sourceSize.x, 1.0/u.sourceSize.y)
                    * float4(FINE_B, FINE_B, 1.0/FINE_B, 1.0/FINE_B);

    float2 uv = (float2(gid) + 0.5) / float2(outW, outH);
    float f = fract(srcSize1.x * uv.x);
    f = 0.5 - f;
    float2 tex = floor(srcSize1.xy * uv) * srcSize1.zw + 0.5 * srcSize1.zw;
    float4 color = float4(0.0);
    float2 dx_ = float2(srcSize1.z, 0.0);

    float invsqrsigma = 1.0 / (2.0 * u.SIGMA_HB * u.SIGMA_HB);
    float wsum = 0.0;

    for (float n = -u.SIZEHB; n <= u.SIZEHB; n += 1.0) {
        float4 pixel = linearize.sample(samp, tex + n * dx_);
        float w_ = exp(-(n + f) * (n + f) * invsqrsigma);
        pixel.a = max(max(pixel.r, pixel.g), pixel.b);
        pixel.a *= pixel.a * pixel.a;
        color += w_ * pixel;
        wsum += w_;
    }
    color /= wsum;
    outTex.write(float4(color.rgb, pow(color.a, 0.333333)), gid);
}

// =========================================================================
// Pass 9 – Bloom Vertical → BloomPass
// =========================================================================

kernel void guestBloomVPass(
    texture2d<float, access::sample>  source   [[texture(0)]],
    texture2d<float, access::write>   outTex   [[texture(1)]],
    constant Uniforms &u                       [[buffer(0)]],
    sampler samp                               [[sampler(0)]],
    uint2 gid                                  [[thread_position_in_grid]])
{
    uint outW = outTex.get_width();
    uint outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float FINE_B = (u.FINE_BLOOM > 0.5) ? u.FINE_BLOOM : mix(0.75, 0.5, -u.FINE_BLOOM);
    float4 srcSize1 = float4(float(outW), u.sourceSize.y, 1.0/float(outW), 1.0/u.sourceSize.y);
    srcSize1 = srcSize1 * float4(FINE_B, FINE_B, 1.0/FINE_B, 1.0/FINE_B);

    float2 uv = (float2(gid) + 0.5) / float2(outW, outH);
    float f = fract(srcSize1.y * uv.y);
    f = 0.5 - f;
    float2 tex = floor(srcSize1.xy * uv) * srcSize1.zw + 0.5 * srcSize1.zw;
    float4 color = float4(0.0);
    float2 dy_ = float2(0.0, srcSize1.w);

    float invsqrsigma = 1.0 / (2.0 * u.SIGMA_VB * u.SIGMA_VB);
    float wsum = 0.0;

    for (float n = -u.SIZEVB; n <= u.SIZEVB; n += 1.0) {
        float4 pixel = source.sample(samp, tex + n * dy_);
        float w_ = exp(-(n + f) * (n + f) * invsqrsigma);
        pixel.a *= pixel.a * pixel.a;
        color += w_ * pixel;
        wsum += w_;
    }
    color /= wsum;
    outTex.write(float4(color.rgb, pow(color.a, 0.175)), gid);
}

// =========================================================================
// Pass 10 – CRT Core (scanlines, beam, curvature, sharpness)
// =========================================================================

// --- Scanline beam functions ---
static inline float st_(float x) { return exp2(-10.0 * x * x); }

static inline float3 sw0(float x, float color, float scanline, float3 c, float beam_min_, float beam_max_, float scans_)
{
    float tmp = mix(beam_min_, beam_max_, color);
    float3 sat = mix(float3(1.0) + scans_, float3(1.0), c);
    float ex = x * tmp;
    ex = ex * ex;
    return exp2(-scanline * ex * sat);
}

static inline float3 sw1(float x, float color, float scanline, float3 c, float beam_min_, float beam_max_, float scans_)
{
    x = mix(x, beam_min_ * x, max(x - 0.4 * color, 0.0));
    float3 sat = mix(float3(1.0) + scans_, float3(1.0), c);
    float tmp = mix(1.2 * beam_min_, beam_max_, color);
    float ex = x * tmp;
    return exp2(-scanline * ex * ex * sat);
}

static inline float3 sw2(float x, float color, float scanline, float3 c, float beam_min_, float beam_max_, float scans_)
{
    float tmp = mix((2.5 - 0.5 * color) * beam_min_, beam_max_, color);
    float3 sat = mix(float3(1.0) + scans_, float3(1.0), c);
    tmp = mix(beam_max_, tmp, pow(x, color + 0.3));
    float ex = x * tmp;
    return exp2(-scanline * ex * ex * sat);
}

static inline float2 Warp(float2 pos, float warpX_, float warpY_, float c_shape_)
{
    pos = pos * 2.0 - 1.0;
    pos = mix(pos,
              float2(pos.x * rsqrt(1.0 - c_shape_ * pos.y * pos.y),
                     pos.y * rsqrt(1.0 - c_shape_ * pos.x * pos.x)),
              float2(warpX_, warpY_) / c_shape_);
    return pos * 0.5 + 0.5;
}

static inline float2 Overscan(float2 pos, float dx_, float dy_)
{
    pos = pos * 2.0 - 1.0;
    pos *= float2(dx_, dy_);
    return pos * 0.5 + 0.5;
}

static inline float3 gc_(float3 c, float gamma_c_, float eps_)
{
    float mc = max(max(c.r, c.g), c.b);
    float mg = pow(mc, 1.0 / gamma_c_);
    return c * mg / (mc + eps_);
}

kernel void guestCrtPass(
    texture2d<float, access::sample>  linearizePass [[texture(0)]],
    texture2d<float, access::sample>  avgLumPass    [[texture(1)]],
    texture2d<float, access::sample>  prePass       [[texture(2)]],
    texture2d<float, access::write>   outTex        [[texture(3)]],
    constant Uniforms &u                            [[buffer(0)]],
    sampler samp                                    [[sampler(0)]],
    uint2 gid                                       [[thread_position_in_grid]])
{
    uint outW = outTex.get_width();
    uint outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float2 OutputSize = float2(outW, outH);
    float2 uv = (float2(gid) + 0.5) / OutputSize;

    float eps_ = 1e-10;
    float scans_ = 1.5 * u.scans;

    // Compute prescale factor
    float2 linTexSize = float2(linearizePass.get_width(), linearizePass.get_height());
    float2 prescalex = linTexSize / u.sourceSize;

    float4 SourceSize = float4(u.sourceSize.x, u.sourceSize.y, 1.0/u.sourceSize.x, 1.0/u.sourceSize.y);
    SourceSize = SourceSize * mix(float4(prescalex.x, 1.0, 1.0/prescalex.x, 1.0),
                                  float4(1.0, prescalex.y, 1.0, 1.0/prescalex.y), u.TATE);

    float lum = avgLumPass.sample(samp, float2(0.5, 0.5)).a;
    float gamma_in = 1.0 / linearizePass.sample(samp, float2(0.25, 0.25)).a;
    float intera = linearizePass.sample(samp, float2(0.75, 0.25)).a;
    bool hscan = (u.hiscan > 0.5);
    bool inter6 = (intera < 0.35 && u.interm == 6.0);
    bool interb = (((intera < 0.35 && !inter6) || (u.no_scanlines > 0.025)) && !hscan);
    bool notate = (u.TATE < 0.5);
    bool vgascan = ((abs(intera - 0.5) < 0.05) && (u.no_scanlines == 0.0));

    float SourceY = mix(SourceSize.y, SourceSize.x, u.TATE);

    float sy = 1.0;
    if (u.intres == 1.0) sy = max(round(SourceY / 224.0), 1.0);
    if (u.intres > 0.25 && u.intres != 1.0) sy = u.intres;
    if (inter6) sy *= 2.0;
    if (vgascan) sy = 0.5; else if (abs(intera - 0.75) < 0.05) sy = 1.0;

    if (notate) SourceSize *= float4(1.0, 1.0/sy, 1.0, sy);
    else        SourceSize *= float4(1.0/sy, 1.0, sy, 1.0);

    // Texcoord calculations
    float2 texcoord = uv;
    if (u.IOS > 0.0 && !interb) {
        float2 ofactor = OutputSize / u.sourceSize;
        float2 intfactor = (u.IOS < 2.5) ? floor(ofactor) : ceil(ofactor);
        float2 diff = ofactor / intfactor;
        float scan = mix(diff.y, diff.x, u.TATE);
        texcoord = Overscan(texcoord, scan, scan);
        if (u.IOS == 1.0 || u.IOS == 3.0)
            texcoord = mix(float2(uv.x, texcoord.y), float2(texcoord.x, uv.y), u.TATE);
    }

    float factor = 1.00 + (1.0 - 0.5 * u.OS) * u.BLOOM / 100.0 - lum * u.BLOOM / 100.0;
    texcoord = Overscan(texcoord, factor, factor);
    texcoord.y -= u.VShift * u.sourceSize.y;  // Use inverted size
    texcoord.y += u.VShift / u.sourceSize.y;  // Correct: subtract VShift * origSize.w
    // Rewrite: just do it like the original
    texcoord.y = (float2(gid).y + 0.5) / OutputSize.y; // re-derive
    // Re-do texcoord properly:
    texcoord = uv;
    if (u.IOS > 0.0 && !interb) {
        float2 ofactor = OutputSize / u.sourceSize;
        float2 intfactor = (u.IOS < 2.5) ? floor(ofactor) : ceil(ofactor);
        float2 diff = ofactor / intfactor;
        float scan = mix(diff.y, diff.x, u.TATE);
        texcoord = Overscan(texcoord, scan, scan);
        if (u.IOS == 1.0 || u.IOS == 3.0)
            texcoord = mix(float2(uv.x, texcoord.y), float2(texcoord.x, uv.y), u.TATE);
    }
    factor = 1.00 + (1.0 - 0.5 * u.OS) * u.BLOOM / 100.0 - lum * u.BLOOM / 100.0;
    texcoord = Overscan(texcoord, factor, factor);
    texcoord.y = texcoord.y - u.VShift * (1.0 / u.sourceSize.y);
    texcoord = Overscan(texcoord,
                        (u.sourceSize.x - u.overscanX) / u.sourceSize.x,
                        (u.sourceSize.y - u.overscanY) / u.sourceSize.y);

    float2 pos = Warp(texcoord, u.warpX, u.warpY, u.c_shape);

    bool smarte = (u.smart_ei > 0.01 && notate);

    float ii = float(inter6) * floor(fmod(float(u.frameCount), 2.0));
    float2 coffset = float2(0.5, 0.5 + 0.5 * ii);

    float2 ps = SourceSize.zw;
    float2 OGL2Pos = pos * SourceSize.xy - coffset;
    float2 fp = fract(OGL2Pos);

    float2 dx = float2(ps.x, 0.0);
    float2 dy = float2(0.0, ps.y);

    float2 offx = dx, off2 = 2.0 * dx;
    float2 offy = dy;
    float fpx = fp.x;
    if (!notate) { offx = dy; off2 = 2.0 * dy; offy = dx; fpx = fp.y; }
    float f = notate ? fp.y : fp.x;

    float2 pC4 = floor(OGL2Pos) * ps + 0.5 * ps;
    pC4.y += float(inter6) * 0.25 * dy.y;

    if ((u.intres == 0.5 && notate && prescalex.y < 1.5) || vgascan)
        pC4.y = floor(pC4.y * u.sourceSize.y) / u.sourceSize.y + 0.5 / u.sourceSize.y;
    if ((u.intres == 0.5 && !notate && prescalex.x < 1.5) || (vgascan && !notate))
        pC4.x = floor(pC4.x * u.sourceSize.x) / u.sourceSize.x + 0.5 / u.sourceSize.x;

    if (interb && u.no_scanlines < 0.025 && !hscan)
        pC4.y = pos.y;
    else if (interb)
        pC4.y = pC4.y + smoothstep(0.40 - 0.5 * u.no_scanlines, 0.60 + 0.5 * u.no_scanlines, f) * mix(SourceSize.w, SourceSize.z, u.TATE);
    if (hscan) pC4 = mix(float2(pC4.x, pos.y), float2(pos.x, pC4.y), u.TATE);

    float zero = exp2(-u.h_sharp);
    float sharp1 = u.s_sharp * zero;

    float idiv = clamp(mix(SourceSize.x, SourceSize.y, u.TATE) / 400.0, 1.0, 2.0);
    float fdivider = max(min(mix(prescalex.x, prescalex.y, u.TATE), 2.0), idiv * float(interb));
    fdivider = 1.0 / max(fdivider, 1.0);

    float wl3 = (2.0 + fpx) * fdivider; wl3 *= wl3; wl3 = exp2(-u.h_sharp * wl3);
    float wl2 = (1.0 + fpx) * fdivider; wl2 *= wl2; wl2 = exp2(-u.h_sharp * wl2);
    float wl1 = (      fpx) * fdivider; wl1 *= wl1; wl1 = exp2(-u.h_sharp * wl1);
    float wr1 = (1.0 - fpx) * fdivider; wr1 *= wr1; wr1 = exp2(-u.h_sharp * wr1);
    float wr2 = (2.0 - fpx) * fdivider; wr2 *= wr2; wr2 = exp2(-u.h_sharp * wr2);
    float wr3 = (3.0 - fpx) * fdivider; wr3 *= wr3; wr3 = exp2(-u.h_sharp * wr3);

    float fp1 = 1.0 - fpx;
    float twl3 = max(wl3 - sharp1, 0.0);
    float twl2 = max(wl2 - sharp1, mix(-0.12, 0.0, 1.0 - fp1 * fp1));
    float twl1 = max(wl1 - sharp1, -0.12);
    float twr1 = max(wr1 - sharp1, -0.12);
    float twr2 = max(wr2 - sharp1, mix(-0.12, 0.0, 1.0 - fpx * fpx));
    float twr3 = max(wr3 - sharp1, 0.0);

    bool sharp = (sharp1 > 0.0);
    float3 c1_, c2_;

    if (smarte) {
        twl3 = 0.0; twr3 = 0.0;
        c1_ = avgLumPass.sample(samp, pC4).xyz;
        c2_ = avgLumPass.sample(samp, pC4 + offy).xyz;
        c1_ = max(c1_ - u.sth, 0.0);
        c2_ = max(c2_ - u.sth, 0.0);
    }

    float3 l3 = linearizePass.sample(samp, pC4 - off2).rgb;
    float3 l2 = linearizePass.sample(samp, pC4 - offx).rgb;
    float3 l1 = linearizePass.sample(samp, pC4).rgb;
    float3 r1 = linearizePass.sample(samp, pC4 + offx).rgb;
    float3 r2 = linearizePass.sample(samp, pC4 + off2).rgb;
    float3 r3 = linearizePass.sample(samp, pC4 + offx + off2).rgb;

    float3 colmin = min(min(l1, r1), min(l2, r2));
    float3 colmax = max(max(l1, r1), max(l2, r2));

    if (smarte) {
        float pc_ = min(u.smart_ei * c1_.y, u.ei_limit);
        float pl_ = min(u.smart_ei * max(c1_.y, c1_.x), u.ei_limit);
        float pr_ = min(u.smart_ei * max(c1_.y, c1_.z), u.ei_limit);
        twl1 = max(wl1 - pc_, 0.01 * wl1); twr1 = max(wr1 - pc_, 0.01 * wr1);
        twl2 = max(wl2 - pl_, 0.01 * wl2); twr2 = max(wr2 - pr_, 0.01 * wr2);
    }

    float3 color1 = (l3*twl3 + l2*twl2 + l1*twl1 + r1*twr1 + r2*twr2 + r3*twr3) /
                     (twl3 + twl2 + twl1 + twr1 + twr2 + twr3);

    if (sharp) color1 = clamp(mix(clamp(color1, colmin, colmax), color1, u.ring), 0.0, 1.0);

    float ts_ = 0.025;
    float3 lumaW = float3(0.2126, 0.7152, 0.0722);
    float lm2 = max(max(l2.r, l2.g), l2.b);
    float lm1 = max(max(l1.r, l1.g), l1.b);
    float rm1 = max(max(r1.r, r1.g), r1.b);
    float rm2 = max(max(r2.r, r2.g), r2.b);

    float swl2_ = max(twl2, 0.0) * (dot(l2, lumaW) + ts_);
    float swl1_ = max(twl1, 0.0) * (dot(l1, lumaW) + ts_);
    float swr1_ = max(twr1, 0.0) * (dot(r1, lumaW) + ts_);
    float swr2_ = max(twr2, 0.0) * (dot(r2, lumaW) + ts_);

    float fscolor1 = (lm2*swl2_ + lm1*swl1_ + rm1*swr1_ + rm2*swr2_) / (swl2_ + swl1_ + swr1_ + swr2_);
    float3 scolor1 = float3(clamp(mix(max(max(color1.r, color1.g), color1.b), fscolor1, u.spike), 0.0, 1.0));

    if (!interb) color1 = pow(color1, float3(u.scangamma / gamma_in));

    float3 color2 = color1;
    float3 scolor2 = scolor1;

    if (!interb && !hscan) {
        pC4 += offy;
        if ((u.intres == 0.5 && notate && prescalex.y < 1.5) || vgascan)
            pC4.y = floor((pos.y + 0.33 * offy.y) * u.sourceSize.y) / u.sourceSize.y + 0.5 / u.sourceSize.y;
        if ((u.intres == 0.5 && !notate && prescalex.x < 1.5) || (vgascan && !notate))
            pC4.x = floor((pos.x + 0.33 * offy.x) * u.sourceSize.x) / u.sourceSize.x + 0.5 / u.sourceSize.x;

        l3 = linearizePass.sample(samp, pC4 - off2).rgb;
        l2 = linearizePass.sample(samp, pC4 - offx).rgb;
        l1 = linearizePass.sample(samp, pC4).rgb;
        r1 = linearizePass.sample(samp, pC4 + offx).rgb;
        r2 = linearizePass.sample(samp, pC4 + off2).rgb;
        r3 = linearizePass.sample(samp, pC4 + offx + off2).rgb;

        colmin = min(min(l1, r1), min(l2, r2));
        colmax = max(max(l1, r1), max(l2, r2));

        if (smarte) {
            float pc_ = min(u.smart_ei * c2_.y, u.ei_limit);
            float pl_ = min(u.smart_ei * max(c2_.y, c2_.x), u.ei_limit);
            float pr_ = min(u.smart_ei * max(c2_.y, c2_.z), u.ei_limit);
            twl1 = max(wl1 - pc_, 0.01 * wl1); twr1 = max(wr1 - pc_, 0.01 * wr1);
            twl2 = max(wl2 - pl_, 0.01 * wl2); twr2 = max(wr2 - pr_, 0.01 * wr2);
        }

        color2 = (l3*twl3 + l2*twl2 + l1*twl1 + r1*twr1 + r2*twr2 + r3*twr3) /
                  (twl3 + twl2 + twl1 + twr1 + twr2 + twr3);
        if (sharp) color2 = clamp(mix(clamp(color2, colmin, colmax), color2, u.ring), 0.0, 1.0);

        lm2 = max(max(l2.r, l2.g), l2.b);
        lm1 = max(max(l1.r, l1.g), l1.b);
        rm1 = max(max(r1.r, r1.g), r1.b);
        rm2 = max(max(r2.r, r2.g), r2.b);
        swl2_ = max(twl2, 0.0) * (dot(l2, lumaW) + ts_);
        swl1_ = max(twl1, 0.0) * (dot(l1, lumaW) + ts_);
        swr1_ = max(twr1, 0.0) * (dot(r1, lumaW) + ts_);
        swr2_ = max(twr2, 0.0) * (dot(r2, lumaW) + ts_);
        float fscolor2 = (lm2*swl2_ + lm1*swl1_ + rm1*swr1_ + rm2*swr2_) / (swl2_ + swl1_ + swr1_ + swr2_);
        scolor2 = float3(clamp(mix(max(max(color2.r, color2.g), color2.b), fscolor2, u.spike), 0.0, 1.0));
        color2 = pow(color2, float3(u.scangamma / gamma_in));
    }

    float3 ctmp = color1;
    float3 sctmp = scolor1;
    float3 color = color1;
    float3 one = float3(1.0);

    if (hscan) { color2 = color1; scolor2 = scolor1; }

    if (!interb) {
        float shape1 = mix(u.scanline1, u.scanline2, f);
        float shape2 = mix(u.scanline1, u.scanline2, 1.0 - f);

        float wt1 = st_(f);
        float wt2 = st_(1.0 - f);

        float3 color00 = color1 * wt1 + color2 * wt2;
        float3 scolor0 = scolor1 * wt1 + scolor2 * wt2;

        ctmp = color00 / (wt1 + wt2);
        sctmp = scolor0 / (wt1 + wt2);

        if (abs(u.rolling_scan) > 0.005) {
            color1 = ctmp; color2 = ctmp;
            scolor1 = sctmp; scolor2 = sctmp;
        }

        float3 cref1 = mix(sctmp, scolor1, u.beam_size);
        float creff1 = pow(max(max(cref1.r, cref1.g), cref1.b), u.scan_falloff);
        float3 cref2 = mix(sctmp, scolor2, u.beam_size);
        float creff2 = pow(max(max(cref2.r, cref2.g), cref2.b), u.scan_falloff);

        if (u.tds > 0.5) {
            shape1 = mix(u.scanline2, shape1, creff1);
            shape2 = mix(u.scanline2, shape2, creff2);
        }

        float scanpix = mix(u.sourceSize.x / OutputSize.x, u.sourceSize.y / OutputSize.y, float(notate));
        float f1 = fract(f - u.rolling_scan * float(u.frameCount) * scanpix);
        float f2 = 1.0 - f1;

        float mc1 = max(max(color1.r, color1.g), color1.b) + eps_;
        float mc2 = max(max(color2.r, color2.g), color2.b) + eps_;
        float3 cref1_ = color1 / mc1;
        float3 cref2_ = color2 / mc2;

        float3 w1, w2;
        if (u.gsl < 0.5) {
            w1 = sw0(f1, creff1, shape1, cref1_, u.beam_min, u.beam_max, scans_);
            w2 = sw0(f2, creff2, shape2, cref2_, u.beam_min, u.beam_max, scans_);
        } else if (u.gsl == 1.0) {
            w1 = sw1(f1, creff1, shape1, cref1_, u.beam_min, u.beam_max, scans_);
            w2 = sw1(f2, creff2, shape2, cref2_, u.beam_min, u.beam_max, scans_);
        } else {
            w1 = sw2(f1, creff1, shape1, cref1_, u.beam_min, u.beam_max, scans_);
            w2 = sw2(f2, creff2, shape2, cref2_, u.beam_min, u.beam_max, scans_);
        }

        float3 w3 = w1 + w2;
        float wf1 = max(max(w3.r, w3.g), w3.b);
        if (wf1 > 1.0) { wf1 = 1.0 / wf1; w1 *= wf1; w2 *= wf1; }

        if (abs(u.clips) > 0.005) {
            float sy_ = mc1; one = (u.clips > 0.0) ? w1 : float3(1.0);
            float sat_ = 1.0001 - min(min(cref1_.r, cref1_.g), cref1_.b);
            color1 = mix(color1, plant(pow(color1, float3(0.70) - 0.325 * sat_), sy_), pow(sat_, 0.3333) * one * abs(u.clips));
            sy_ = mc2; one = (u.clips > 0.0) ? w2 : float3(1.0);
            sat_ = 1.0001 - min(min(cref2_.r, cref2_.g), cref2_.b);
            color2 = mix(color2, plant(pow(color2, float3(0.70) - 0.325 * sat_), sy_), pow(sat_, 0.3333) * one * abs(u.clips));
        }

        color = gc_(color1, u.gamma_c, eps_) * w1 + gc_(color2, u.gamma_c, eps_) * w2;
        color = min(color, 1.0);
    }

    if (interb) color = gc_(color1, u.gamma_c, eps_);

    float colmx = max(max(ctmp.r, ctmp.g), ctmp.b);
    if (!interb) color = pow(color, float3(gamma_in / u.scangamma));

    outTex.write(float4(color, colmx), gid);
}

// =========================================================================
// Pass 11 – Deconvergence / Final Composite
// =========================================================================

// Shadow mask function (supports 15 types, -1 to 14)
static float3 MaskFn(float2 pos, float mx, float mb, float shadowMask_, float maskstr_,
                     float mcut_, float maskDark_, float maskLight_, float maskboost_, float mask_layout_)
{
    float3 mask = float3(maskDark_);
    float3 one = float3(1.0);

    if (shadowMask_ == 0.0) {
        float mc = 1.0 - max(maskstr_, 0.0);
        pos.x = fract(pos.x * 0.5);
        if (pos.x < 0.49) { mask.r = 1.0; mask.g = mc; mask.b = 1.0; }
        else { mask.r = mc; mask.g = 1.0; mask.b = mc; }
    }
    else if (shadowMask_ == 1.0) {
        float line = maskLight_;
        float odd = 0.0;
        if (fract(pos.x / 6.0) < 0.49) odd = 1.0;
        if (fract((pos.y + odd) / 2.0) < 0.49) line = maskDark_;
        pos.x = floor(fmod(pos.x, 3.0));
        if      (pos.x < 0.5) mask.r = maskLight_;
        else if (pos.x < 1.5) mask.g = maskLight_;
        else                   mask.b = maskLight_;
        mask *= line;
    }
    else if (shadowMask_ == 2.0) {
        pos.x = floor(fmod(pos.x, 3.0));
        if      (pos.x < 0.5) mask.r = maskLight_;
        else if (pos.x < 1.5) mask.g = maskLight_;
        else                   mask.b = maskLight_;
    }
    else if (shadowMask_ == 3.0) {
        pos.x += pos.y * 3.0;
        pos.x = fract(pos.x / 6.0);
        if      (pos.x < 0.3) mask.r = maskLight_;
        else if (pos.x < 0.6) mask.g = maskLight_;
        else                   mask.b = maskLight_;
    }
    else if (shadowMask_ == 4.0) {
        pos.xy = floor(pos.xy * float2(1.0, 0.5));
        pos.x += pos.y * 3.0;
        pos.x = fract(pos.x / 6.0);
        if      (pos.x < 0.3) mask.r = maskLight_;
        else if (pos.x < 0.6) mask.g = maskLight_;
        else                   mask.b = maskLight_;
    }
    else if (shadowMask_ == 5.0) {
        mask = float3(0.0);
        pos.x = fract(pos.x / 2.0);
        if (pos.x < 0.49) { mask.r = 1.0; mask.b = 1.0; }
        else mask.g = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 6.0) {
        mask = float3(0.0);
        pos.x = floor(fmod(pos.x, 3.0));
        if      (pos.x < 0.5) mask.r = 1.0;
        else if (pos.x < 1.5) mask.g = 1.0;
        else                   mask.b = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 7.0) {
        mask = float3(0.0);
        pos.x = fract(pos.x / 2.0);
        if (pos.x < 0.49) mask = float3(0.0); else mask = float3(1.0);
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 8.0) {
        mask = float3(0.0);
        pos.x = fract(pos.x / 3.0);
        if      (pos.x < 0.3) mask = float3(0.0);
        else if (pos.x < 0.6) mask = float3(1.0);
        else                   mask = float3(1.0);
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 9.0) {
        mask = float3(0.0);
        pos.x = fract(pos.x / 3.0);
        if      (pos.x < 0.3) mask = float3(0.0);
        else if (pos.x < 0.6) { mask.r = 1.0; mask.b = 1.0; }
        else                   mask.g = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 10.0) {
        mask = float3(0.0);
        pos.x = fract(pos.x * 0.25);
        if      (pos.x < 0.2) mask = float3(0.0);
        else if (pos.x < 0.4) mask.r = 1.0;
        else if (pos.x < 0.7) mask.g = 1.0;
        else                   mask.b = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 11.0) {
        mask = float3(0.0);
        pos.x = fract(pos.x * 0.25);
        if      (pos.x < 0.2) mask.r = 1.0;
        else if (pos.x < 0.4) { mask.r = 1.0; mask.g = 1.0; }
        else if (pos.x < 0.7) { mask.g = 1.0; mask.b = 1.0; }
        else                   mask.b = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 12.0) {
        mask = float3(0.0);
        pos.x = floor(fmod(pos.x, 7.0));
        if      (pos.x < 0.5) mask = float3(0.0);
        else if (pos.x < 2.5) mask.r = 1.0;
        else if (pos.x < 4.5) mask.g = 1.0;
        else                   mask.b = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else if (shadowMask_ == 13.0) {
        mask = float3(0.0);
        pos.x = floor(fmod(pos.x, 6.0));
        if      (pos.x < 0.5) mask = float3(0.0);
        else if (pos.x < 1.5) mask.r = 1.0;
        else if (pos.x < 2.5) { mask.r = 1.0; mask.g = 1.0; }
        else if (pos.x < 3.5) mask = float3(1.0);
        else if (pos.x < 4.5) { mask.g = 1.0; mask.b = 1.0; }
        else                   mask.b = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }
    else {
        // shadowMask == 14 or any other
        mask = float3(0.0);
        pos.x = floor(fmod(pos.x, 5.0));
        if      (pos.x < 0.5) mask = float3(0.0);
        else if (pos.x < 1.5) mask.r = 1.0;
        else if (pos.x < 2.5) { mask.r = 1.0; mask.g = 1.0; }
        else if (pos.x < 3.5) { mask.g = 1.0; mask.b = 1.0; }
        else                   mask.b = 1.0;
        mask = clamp(mix(mix(one, mask, mcut_), mix(one, mask, maskstr_), mx), 0.0, 1.0);
    }

    if (mask_layout_ > 0.5) mask = mask.rbg;
    float maskmin = min(min(mask.r, mask.g), mask.b);
    return (mask - maskmin) * (1.0 + (maskboost_ - 1.0) * mb) + maskmin;
}

static float SlotMaskFn(float2 pos, float m, float swidth, float slotmask_, float slotmask1_,
                        float slotms_, float double_slot_)
{
    if ((slotmask_ + slotmask1_) == 0.0) return 1.0;
    pos.y = floor(pos.y / slotms_);
    float mlen = swidth * 2.0;
    float px = floor(fmod(pos.x, 0.99999 * mlen));
    float py = floor(fract(pos.y / (2.0 * double_slot_)) * 2.0 * double_slot_);
    float slot_dark = mix(1.0 - slotmask1_, 1.0 - slotmask_, m);
    float slot = 1.0;
    if (py == 0.0 && px < swidth) slot = slot_dark;
    else if (py == double_slot_ && px >= swidth) slot = slot_dark;
    return slot;
}

static float humbar_(float pos, float barintensity_, float barspeed_, uint frameCount)
{
    if (barintensity_ == 0.0) return 1.0;
    pos = (barintensity_ >= 0.0) ? pos : (1.0 - pos);
    pos = fract(pos + fmod(float(frameCount), barspeed_) / (barspeed_ - 1.0));
    pos = (barintensity_ < 0.0) ? pos : (1.0 - pos);
    return (1.0 - barintensity_) + barintensity_ * pos;
}

static float corner_(float2 pos, float2 OutputSize, float csize_, float bsize1_, float sborder_)
{
    pos = abs(2.0 * (pos - 0.5));
    float2 aspect = float2(1.0, OutputSize.x / OutputSize.y);
    float b = bsize1_ * 0.05 + 0.0005;
    pos.y = pos.y + b * (aspect.y - 1.0);
    float2 crn = max(float2(csize_), 2.0 * b + 0.0015);
    float2 cp = max(pos - (1.0 - crn * aspect), 0.0) / aspect;
    float cd = sqrt(dot(cp, cp));
    pos = max(pos, 1.0 - crn + cd);
    float res = mix(1.0, 0.0, smoothstep(1.0 - b, 1.0, sqrt(max(pos.x, pos.y))));
    return pow(res, sborder_);
}

static float3 declip_(float3 c, float b)
{
    float m = max(max(c.r, c.g), c.b);
    if (m > b) c = c * b / m;
    return c;
}

static float igc_(float mc, float gamma_c_) { return pow(mc, gamma_c_); }

static float3 gc2_(float3 c, float w3, float gamma_c2_, float eps_)
{
    float mc = max(max(c.r, c.g), c.b);
    float gp = 1.0 / (1.0 + (gamma_c2_ - 1.0) * mix(0.375, 1.0, w3));
    float mg = pow(mc, gp);
    return c * mg / (mc + eps_);
}

static float3 noiseFn(float3 v, float addnoised_)
{
    if (addnoised_ < 0.0) v.z = -addnoised_; else v.z = fmod(v.z, 6001.0) / 1753.0;
    v = fract(v) + fract(v * 1e4) + fract(v * 1e-4);
    v += float3(0.12345, 0.6789, 0.314159);
    v = fract(v * dot(v, v) * 123.456);
    v = fract(v * dot(v, v) * 123.456);
    v = fract(v * dot(v, v) * 123.456);
    v = fract(v * dot(v, v) * 123.456);
    return v;
}

kernel void guestDeconvergencePass(
    texture2d<float, access::sample>  linearizePass [[texture(0)]],
    texture2d<float, access::sample>  avgLumPass    [[texture(1)]],
    texture2d<float, access::sample>  glowPass      [[texture(2)]],
    texture2d<float, access::sample>  bloomPass     [[texture(3)]],
    texture2d<float, access::sample>  prePass       [[texture(4)]],
    texture2d<float, access::sample>  crtSource     [[texture(5)]],  // output of CRT pass
    texture2d<float, access::sample>  stockPass     [[texture(6)]],
    texture2d<float, access::write>   outTex        [[texture(7)]],
    constant Uniforms &u                            [[buffer(0)]],
    sampler samp                                    [[sampler(0)]],
    uint2 gid                                       [[thread_position_in_grid]])
{
    uint outW = outTex.get_width();
    uint outH = outTex.get_height();
    if (gid.x >= outW || gid.y >= outH) return;

    float eps_ = 1e-10;
    float2 OutputSize = float2(outW, outH);
    float2 uv = (float2(gid) + 0.5) / OutputSize;

    float lum = avgLumPass.sample(samp, float2(0.5, 0.5)).a;
    float gamma_in = 1.0 / linearizePass.sample(samp, float2(0.25, 0.25)).a;
    float intera = linearizePass.sample(samp, float2(0.75, 0.25)).a;
    bool inter6 = (intera < 0.35 && u.interm == 6.0);
    bool interb = (((intera < 0.35 && !inter6) || u.no_scanlines > 0.025) && u.hiscan < 0.5);
    bool notate = (u.TATE < 0.5);

    // Texcoord calculations (same as CRT pass)
    float2 texcoord = uv;
    if (u.IOS > 0.0 && !interb) {
        float2 ofactor = OutputSize / u.sourceSize;
        float2 intfactor = (u.IOS < 2.5) ? floor(ofactor) : ceil(ofactor);
        float2 diff = ofactor / intfactor;
        float scan = mix(diff.y, diff.x, u.TATE);
        texcoord = Overscan(texcoord, scan, scan);
        if (u.IOS == 1.0 || u.IOS == 3.0)
            texcoord = mix(float2(uv.x, texcoord.y), float2(texcoord.x, uv.y), u.TATE);
    }
    float factor = 1.00 + (1.0 - 0.5 * u.OS) * u.BLOOM / 100.0 - lum * u.BLOOM / 100.0;
    texcoord = Overscan(texcoord, factor, factor);
    texcoord.y = texcoord.y - u.VShift * (1.0 / u.sourceSize.y);
    texcoord = Overscan(texcoord,
                        (u.sourceSize.x - u.overscanX) / u.sourceSize.x,
                        (u.sourceSize.y - u.overscanY) / u.sourceSize.y);

    float2 pos1 = uv;
    float2 pos  = Warp(texcoord, u.warpX, u.warpY, u.c_shape);
    float2 pos0 = Warp(uv, u.warpX, u.warpY, u.c_shape);
    float2 posb = 2.0 * (pos0 - 0.5);
    if (u.BLOOM < 0.5) posb = max(abs(posb), abs(2.0 * (pos - 0.5))) * sign(posb);
    posb = 0.5 * posb + 0.5;

    // Fetch colour and bloom/glow
    float3 color = crtSource.sample(samp, pos1).rgb;
    float3 Bloom = bloomPass.sample(samp, pos).rgb;
    float3 Glow  = glowPass.sample(samp, pos).rgb;

    // Deconvergence
    if ((abs(u.deconrr) + abs(u.deconrry) + abs(u.deconrg) + abs(u.deconrgy) + abs(u.deconrb) + abs(u.deconrby)) > 0.2) {
        float stepx = 1.0 / OutputSize.x;
        float stepy = 1.0 / OutputSize.y;
        float2 dx_ = float2(stepx, 0.0);
        float2 dy_ = float2(0.0, stepy);
        float posx = 2.0 * pos1.x - 1.0;
        float posy = 2.0 * pos1.y - 1.0;
        if (u.dctypex > 0.025) { posx = sign(posx) * pow(abs(posx), 1.05 - u.dctypex); dx_ = posx * dx_; }
        if (u.dctypey > 0.025) { posy = sign(posy) * pow(abs(posy), 1.05 - u.dctypey); dy_ = posy * dy_; }

        float2 rc = u.deconrr * dx_ + u.deconrry * dy_;
        float2 gc_ = u.deconrg * dx_ + u.deconrgy * dy_;
        float2 bc = u.deconrb * dx_ + u.deconrby * dy_;

        float r1 = crtSource.sample(samp, pos1 + rc).r;
        float g1 = crtSource.sample(samp, pos1 + gc_).g;
        float b1 = crtSource.sample(samp, pos1 + bc).b;
        color = clamp(mix(color, float3(r1, g1, b1), u.decons), 0.0, 1.0);

        r1 = bloomPass.sample(samp, pos + rc).r;
        g1 = bloomPass.sample(samp, pos + gc_).g;
        b1 = bloomPass.sample(samp, pos + bc).b;
        float3 bd = float3(r1, g1, b1);
        Bloom = mix(Bloom, bd, min(u.decons, 1.0));
        Glow = Bloom;

        r1 = glowPass.sample(samp, pos + rc).r;
        g1 = glowPass.sample(samp, pos + gc_).g;
        b1 = glowPass.sample(samp, pos + bc).b;
        Glow = mix(Glow, float3(r1, g1, b1), min(u.decons, 1.0));
    }

    float cm = igc_(max(max(color.r, color.g), color.b), u.gamma_c);
    float mx1 = crtSource.sample(samp, pos1).a;
    float colmx = max(mx1, cm);
    float w3 = min((max((cm - 0.0005) * 1.0005, 0.0) + 0.0001) / (colmx + 0.0005), 1.0);
    if (interb) w3 = 1.0;

    float2 dx = mix(float2(0.001, 0.0), float2(0.0, 0.001), u.TATE);
    float mx0 = crtSource.sample(samp, pos1 - dx).a;
    float mx2 = crtSource.sample(samp, pos1 + dx).a;
    float mxg = max(max(mx0, mx1), max(mx2, cm));
    float mx = pow(mxg, 1.40 / gamma_in);
    float cx = pow(colmx, 1.4 / gamma_in);

    // Mask boost tweak
    dx = mix(float2(u.sourceSize.x > 0 ? 1.0/u.sourceSize.x : 0, 0.0),
             float2(0.0, u.sourceSize.y > 0 ? 1.0/u.sourceSize.y : 0), u.TATE) * 0.25;
    mx0 = crtSource.sample(samp, pos1 - dx).a;
    mx2 = crtSource.sample(samp, pos1 + dx).a;
    float mb = (1.0 - min(abs(mx0 - mx2) / (0.5 + mx1), 1.0));

    float3 one = float3(1.0);
    float3 orig1 = color;
    float3 cmask = one;
    float3 cmask1 = one;
    float3 cmask2 = one;

    // Mask widths
    float mwidths[15] = {2.0, 3.0, 3.0, 6.0, 6.0, 2.4, 3.4, 2.4, 3.25, 3.4, 4.4, 4.25, 7.4, 6.25, 5.25};
    int maskIdx = clamp(int(u.shadowMask), 0, 14);
    float mwidth = mwidths[maskIdx];
    float mask_compensate = fract(mwidth);
    float mwidth1 = mwidth;

    if (u.shadowMask > -0.5) {
        float2 maskcoord = float2(gid.yx);
        if (notate) maskcoord = float2(gid.xy);  // Original uses yx then flips for notate
        // Actually: gl_FragCoord = vTexCoord * OutputSize, and maskcoord = gl_FragCoord.yx, then if notate: .yx
        // In Metal compute, gid = thread position = pixel position
        maskcoord = float2(float(gid.x), float(gid.y));
        float2 scoord = maskcoord;

        mwidth = floor(mwidth) * u.masksize;
        float swidth = mwidth;
        bool zoomed = (abs(u.mask_zoom) > 0.75);
        float mscale = 1.0;
        float2 maskcoord0 = maskcoord;
        maskcoord.y = floor(maskcoord.y / u.masksize);
        mwidth1 = max(mwidth + u.mask_zoom, 2.0);

        if (u.mshift > 0.25) {
            float stagg_lvl = 1.0;
            if (fract(u.mshift) > 0.25) stagg_lvl = 2.0;
            float next_line = float(floor(fmod(maskcoord.y, 2.0 * stagg_lvl)) < stagg_lvl);
            maskcoord0.x = maskcoord0.x + next_line * 0.5 * mwidth1;
        }
        maskcoord = maskcoord0 / u.masksize;

        if (!zoomed) {
            cmask = MaskFn(floor(maskcoord), mx, mb, u.shadowMask, u.maskstr, u.mcut, u.maskDark, u.maskLight, u.maskboost, u.mask_layout);
        } else {
            mscale = mwidth1 / mwidth;
            float mlerp = fract(maskcoord.x / mscale);
            if (u.mzoom_sh > 0.025) mlerp = clamp((1.0 + u.mzoom_sh) * mlerp - 0.5 * u.mzoom_sh, 0.0, 1.0);
            float mcoord = floor(maskcoord.x / mscale);
            if (u.shadowMask == 12.0 && u.mask_zoom == -2.0) mcoord = ceil(maskcoord.x / mscale);
            cmask = mix(MaskFn(float2(mcoord, maskcoord.y), mx, mb, u.shadowMask, u.maskstr, u.mcut, u.maskDark, u.maskLight, u.maskboost, u.mask_layout),
                       MaskFn(float2(mcoord + 1.0, maskcoord.y), mx, mb, u.shadowMask, u.maskstr, u.mcut, u.maskDark, u.maskLight, u.maskboost, u.mask_layout),
                       mlerp);
        }

        float sm_offset = 0.0;
        bool bsm_offset = (u.shadowMask == 0.0 || u.shadowMask == 2.0 || u.shadowMask == 5.0 ||
                           u.shadowMask == 6.0 || u.shadowMask == 8.0 || u.shadowMask == 11.0);
        if (zoomed) { if (u.mask_layout < 0.5 && bsm_offset) sm_offset = 1.0; else if (bsm_offset) sm_offset = -1.0; }

        swidth = round(mwidth1);
        if (u.slotwidth > 0.5) swidth = u.slotwidth;
        float smask = SlotMaskFn(scoord + float2(sm_offset, 0.0), mx, swidth, u.slotmask, u.slotmask1, u.slotms, u.double_slot);
        smask = clamp(smask + mix(u.smask_mit, 0.0, w3 * pow(colmx, 0.3)), 0.0, 1.0);

        cmask2 = cmask;
        cmask *= smask;
        cmask1 = cmask;

        if (abs(u.mask_bloom) > 0.025) {
            float maxbl = max(max(max(Bloom.r, Bloom.g), Bloom.b), mxg);
            maxbl = maxbl * max(mix(1.0, 2.0 - colmx, u.bloom_dist), 0.0);
            if (u.mask_bloom > 0.025)
                cmask = max(min(cmask + maxbl * u.mask_bloom, 1.0), cmask);
            else
                cmask = max(mix(cmask, cmask * (1.0 - 0.5 * maxbl) + plant(pow(Bloom, float3(0.35)), maxbl), -u.mask_bloom), cmask);
        }

        color = pow(color, float3(u.mask_gamma / gamma_in));
        color = color * cmask;
        color = min(color, 1.0);
        color = pow(color, float3(gamma_in / u.mask_gamma));
        cmask = min(cmask, 1.0);
        cmask1 = min(cmask1, 1.0);
    }

    float dark_compensate = mix(max(clamp(mix(u.mcut, u.maskstr, mx), 0.0, 1.0) - 1.0 + mask_compensate, 0.0) + 1.0, 1.0, mx);
    if (u.shadowMask < -0.5) dark_compensate = 1.0;
    float bb = mix(u.brightboost, u.brightboost1, mx) * dark_compensate;
    color *= bb;

    float3 Ref = linearizePass.sample(samp, pos).rgb;
    float maxb = bloomPass.sample(samp, pos).a;
    float2 preUV = clamp(pos, 0.0 + 0.5 / u.sourceSize, 1.0 - 0.5 / u.sourceSize);
    float vig = prePass.sample(samp, preUV).a;

    if (u.pr_scan > 0.025) {
        float mbl = max(max(Bloom.r, Bloom.g), Bloom.b);
        Bloom = mix(Bloom, mix(Bloom, plant(orig1, mbl), min(2.5 * (1.0 - w3), 1.0)), min(2.0 * u.pr_scan, 1.0));
    }

    float3 Bloom1 = Bloom;
    float3 bcmask = mix(one, cmask1, u.bmask1);
    float3 hcmask = mix(one, cmask1, u.hmask1);

    // Bloom compositing
    if (abs(u.bloom) > 0.025) {
        if (u.bloom < -0.01) Bloom1 = plant(Bloom, maxb);
        Bloom1 = min(Bloom1 * (orig1 + color), max(0.5 * (colmx + orig1 - color), 0.001 * Bloom1));
        Bloom1 = 0.5 * (Bloom1 + mix(Bloom1, mix(colmx * orig1, Bloom1, 0.5), 1.0 - color));
        Bloom1 = bcmask * Bloom1 * max(mix(1.0, 2.0 - colmx, u.bloom_dist), 0.0);
        color = pow(pow(color, float3(u.mask_gamma / gamma_in)) + abs(u.bloom) * pow(Bloom1, float3(u.mask_gamma / gamma_in)), float3(gamma_in / u.mask_gamma));
    }

    // Halation
    if (u.halation > 0.01) {
        Bloom = 0.5 * (Bloom + Bloom * Bloom);
        float mbl = max(max(Bloom.r, Bloom.g), Bloom.b);
        float cmxh = 0.5 * (colmx + colmx * colmx);
        mbl = mix(mix(cmxh, mix(cmxh, mbl, mbl), colmx), mbl, mb);
        Bloom = plant(Bloom, mix(sqrt(mbl * cmxh), max((mbl - 0.15 * (1.0 - colmx)), 0.4 * cmxh), pow(colmx, 0.25))) * mix(0.425, 1.0, colmx);
        Bloom = (3.0 - colmx - color) * plant(0.325 + orig1 / w3, 0.5 * (1.0 + w3)) * hcmask * Bloom;
        color = pow(pow(color, float3(u.mask_gamma / gamma_in)) + u.halation * pow(Bloom, float3(u.mask_gamma / gamma_in)), float3(gamma_in / u.mask_gamma));
    } else if (u.halation < -0.01) {
        float mbl = max(max(Bloom.r, Bloom.g), Bloom.b);
        Bloom = plant(Bloom + Ref + orig1 + Bloom * Bloom * Bloom, min(mbl * mbl, 0.75));
        Bloom = 2.0 * mix(1.0, w3, 0.5 * colmx) * hcmask * Bloom;
        color = color - u.halation * Bloom;
    }

    color = min(color, 1.0);
    color = gc2_(color, w3, u.gamma_c2, eps_);

    // Smooth mask
    if (u.smoothmask > 0.125) {
        float w4 = pow(w3, 0.425 + 0.3 * u.smoothmask);
        w4 = max(w4 - 0.175 * colmx * u.smoothmask, 0.2);
        color = mix(min(color / w4, plant(orig1, 1.0 + 0.175 * colmx * u.smoothmask)) * w4, color, w4);
    }

    // Glow compositing
    if (u.m_glow < 0.5) {
        Glow = mix(Glow, 0.25 * color, colmx);
    } else {
        float maxb_ = max(max(Glow.r, Glow.g), Glow.b);
        float3 orig2 = plant(orig1 + 0.001 * Ref, 1.0);
        float3 BloomG = plant(Glow, 1.0);
        float3 RefG = abs(orig2 - BloomG);
        float mx0_ = max(max(orig2.r, orig2.g), orig2.b) - min(min(orig2.r, orig2.g), orig2.b);
        float mx2_ = max(max(BloomG.r, BloomG.g), BloomG.b) - min(min(BloomG.r, BloomG.g), BloomG.b);
        BloomG = mix(maxb_ * min(BloomG, orig2),
                     mix(mix(Glow, max(max(RefG.r, RefG.g), RefG.b) * Glow, max(mx, mx0_)),
                         mix(color, Glow, mx2_), max(mx0_, mx2_) * RefG),
                     min(sqrt((1.10 - mx0_) * (0.10 + mx2_)), 1.0));
        if (u.m_glow > 1.5) Glow = mix(0.5 * Glow * Glow, BloomG, BloomG);
        Glow = mix(u.m_glow_low * Glow, u.m_glow_high * BloomG, pow(colmx, u.m_glow_dist / gamma_in));
    }

    if (u.m_glow < 0.5) {
        if (u.glow >= 0.0) color = color + 0.5 * Glow * u.glow;
        else color = color + abs(u.glow) * min(cmask2 * cmask2, 1.0) * Glow;
    } else {
        float3 cmaskg = clamp(mix(one, cmask1, u.m_glow_mask), 0.0, 1.0);
        color = color + abs(u.glow) * cmaskg * Glow;
    }

    color = min(color, 1.0);

    // Edge mask
    if (u.edgemask > 0.05) {
        dx = mix(float2(0.001, 0.0), float2(0.0, 0.001), u.TATE);
        mx0 = crtSource.sample(samp, pos1 - dx).a;
        mx0 = crtSource.sample(samp, pos1 - dx * (1.0 - 0.75 * sqrt(mx0))).a;
        mx2 = crtSource.sample(samp, pos1 + dx).a;
        mx2 = crtSource.sample(samp, pos1 + dx * (1.0 - 0.75 * sqrt(mx2))).a;
        float mx3 = crtSource.sample(samp, pos1 - 4.0 * dx).a;
        float mx4 = crtSource.sample(samp, pos1 + 4.0 * dx).a;
        mx4 = max(pow(abs(mx3 - mx4), 0.55 - 0.40 * cx), min(max(mx3, mx4) / min(0.1 + cx, 1.0), 1.0));
        mb = (1.0 - abs(pow(mx0, 1.0 - 0.65 * mx2) - pow(mx2, 1.0 - 0.65 * mx0)));
        mb = mx4 * u.edgemask * (1.0001 - mb * mb);
        float3 temp = mix(color, orig1, mb);
        color = max(temp + mix(3.5 * mb * mix(1.625 * temp, temp, cx), float3(0.0), pow(color, float3(0.75) - 0.5 * colmx)), color);
    }

    color = color * mix(1.0, mix(0.5 * (1.0 + w3), w3, mx), u.pr_scan);  // preserve scanlines
    color = min(color, max(orig1, color) * mix(one, cmask1, u.mclip));     // preserve mask

    color = pow(color, float3(1.0 / u.gamma_out));

    // Noise
    float rc = 0.6 * sqrt(max(max(color.r, color.g), color.b)) + 0.4;
    if (abs(u.addnoised) > 0.01) {
        float3 noise0 = noiseFn(float3(floor(OutputSize * uv / u.noiseresd), float(u.frameCount)), u.addnoised);
        if (u.noisetype < 0.5)
            color = mix(color, noise0, 0.25 * abs(u.addnoised) * rc);
        else
            color = min(color * mix(1.0, 1.5 * noise0.x, 0.5 * abs(u.addnoised)), 1.0);
    }

    // Base mask
    colmx = max(max(orig1.r, orig1.g), orig1.b);
    color = color + u.bmask * mix(cmask2, 0.125 * (1.0 - colmx) * color, min(20.0 * colmx, 1.0));

    // Final: vignette, humbar, post brightness, corners, original image overlay
    float4 finalColor = float4(color * vig *
                               humbar_(mix(pos.y, pos.x, u.bardir), u.barintensity, u.barspeed, u.frameCount) *
                               u.post_br *
                               corner_(posb, OutputSize, u.csize, u.bsize1, u.sborder), 1.0);

    // Original image overlay
    if (u.oimage > 0.0) {
        float2 stockUV = (floor(pos1 * u.sourceSize) + 0.5) / u.sourceSize;
        float4 stockColor = stockPass.sample(samp, stockUV);
        if (pos1.x < u.oimage - 0.00025) finalColor = stockColor;
    }

    outTex.write(finalColor, gid);
}

} // namespace crtguest
