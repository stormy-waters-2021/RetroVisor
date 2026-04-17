// -----------------------------------------------------------------------------
// This file is part of RetroVisor
//
// MSL port of crt-lottes by Timothy Lottes (public domain)
// Ported from: https://github.com/libretro/slang-shaders/blob/master/crt/shaders/crt-lottes.slang
//
// This is more along the style of a really good CGA arcade monitor.
// With RGB inputs instead of NTSC.
// The shadow mask example has the mask rotated 90 degrees for less
// chromatic aberration.
// -----------------------------------------------------------------------------

#include <metal_stdlib>

using namespace metal;

namespace crtlottes {

    struct Uniforms {

        float hardScan;
        float hardPix;
        float warpX;
        float warpY;
        float maskDark;
        float maskLight;
        float scaleInLinearGamma;
        float shadowMask;
        float brightBoost;
        float hardBloomScan;
        float hardBloomPix;
        float bloomAmount;
        float shape;

        float2 sourceSize;   // input texture dimensions
        float2 outputSize;   // output texture dimensions
    };

    // sRGB to Linear
    inline float ToLinear1(float c, float scaleLinear) {
        if (scaleLinear == 0.0) return c;
        return (c <= 0.04045) ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
    }

    inline float3 ToLinear(float3 c, float scaleLinear) {
        if (scaleLinear == 0.0) return c;
        return float3(ToLinear1(c.r, scaleLinear),
                      ToLinear1(c.g, scaleLinear),
                      ToLinear1(c.b, scaleLinear));
    }

    // Linear to sRGB
    inline float ToSrgb1(float c, float scaleLinear) {
        if (scaleLinear == 0.0) return c;
        return (c < 0.0031308) ? c * 12.92 : 1.055 * pow(c, 0.41666) - 0.055;
    }

    inline float3 ToSrgb(float3 c, float scaleLinear) {
        if (scaleLinear == 0.0) return c;
        return float3(ToSrgb1(c.r, scaleLinear),
                      ToSrgb1(c.g, scaleLinear),
                      ToSrgb1(c.b, scaleLinear));
    }

    // Nearest emulated sample given floating point position and texel offset.
    inline float3 Fetch(float2 pos, float2 off,
                        float2 sourceSize, float brightBoost, float scaleLinear,
                        texture2d<float, access::sample> src, sampler sam) {
        pos = (floor(pos * sourceSize + off) + float2(0.5)) / sourceSize;
        return ToLinear(brightBoost * src.sample(sam, pos).rgb, scaleLinear);
    }

    // Distance in emulated pixels to nearest texel.
    inline float2 Dist(float2 pos, float2 sourceSize) {
        pos = pos * sourceSize;
        return -((pos - floor(pos)) - float2(0.5));
    }

    // 1D Gaussian.
    inline float Gaus(float pos, float scale, float shape) {
        return exp2(scale * pow(abs(pos), shape));
    }

    // 3-tap Gaussian filter along horz line.
    inline float3 Horz3(float2 pos, float off, constant Uniforms &u,
                        texture2d<float, access::sample> src, sampler sam) {
        float3 b = Fetch(pos, float2(-1.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 c = Fetch(pos, float2( 0.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 d = Fetch(pos, float2( 1.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float dst = Dist(pos, u.sourceSize).x;

        float scale = u.hardPix;
        float wb = Gaus(dst - 1.0, scale, u.shape);
        float wc = Gaus(dst + 0.0, scale, u.shape);
        float wd = Gaus(dst + 1.0, scale, u.shape);

        return (b * wb + c * wc + d * wd) / (wb + wc + wd);
    }

    // 5-tap Gaussian filter along horz line.
    inline float3 Horz5(float2 pos, float off, constant Uniforms &u,
                        texture2d<float, access::sample> src, sampler sam) {
        float3 a = Fetch(pos, float2(-2.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 b = Fetch(pos, float2(-1.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 c = Fetch(pos, float2( 0.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 d = Fetch(pos, float2( 1.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 e = Fetch(pos, float2( 2.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float dst = Dist(pos, u.sourceSize).x;

        float scale = u.hardPix;
        float wa = Gaus(dst - 2.0, scale, u.shape);
        float wb = Gaus(dst - 1.0, scale, u.shape);
        float wc = Gaus(dst + 0.0, scale, u.shape);
        float wd = Gaus(dst + 1.0, scale, u.shape);
        float we = Gaus(dst + 2.0, scale, u.shape);

        return (a * wa + b * wb + c * wc + d * wd + e * we) / (wa + wb + wc + wd + we);
    }

    // 7-tap Gaussian filter along horz line (for bloom).
    inline float3 Horz7(float2 pos, float off, constant Uniforms &u,
                        texture2d<float, access::sample> src, sampler sam) {
        float3 a = Fetch(pos, float2(-3.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 b = Fetch(pos, float2(-2.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 c = Fetch(pos, float2(-1.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 d = Fetch(pos, float2( 0.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 e = Fetch(pos, float2( 1.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 f = Fetch(pos, float2( 2.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float3 g = Fetch(pos, float2( 3.0, off), u.sourceSize, u.brightBoost, u.scaleInLinearGamma, src, sam);
        float dst = Dist(pos, u.sourceSize).x;

        float scale = u.hardBloomPix;
        float wa = Gaus(dst - 3.0, scale, u.shape);
        float wb = Gaus(dst - 2.0, scale, u.shape);
        float wc = Gaus(dst - 1.0, scale, u.shape);
        float wd = Gaus(dst + 0.0, scale, u.shape);
        float we = Gaus(dst + 1.0, scale, u.shape);
        float wf = Gaus(dst + 2.0, scale, u.shape);
        float wg = Gaus(dst + 3.0, scale, u.shape);

        return (a * wa + b * wb + c * wc + d * wd + e * we + f * wf + g * wg)
             / (wa + wb + wc + wd + we + wf + wg);
    }

    // Return scanline weight.
    inline float Scan(float2 pos, float off, float hardScan, float2 sourceSize, float shape) {
        float dst = Dist(pos, sourceSize).y;
        return Gaus(dst + off, hardScan, shape);
    }

    // Return scanline weight for bloom.
    inline float BloomScan(float2 pos, float off, float hardBloomScan, float2 sourceSize, float shape) {
        float dst = Dist(pos, sourceSize).y;
        return Gaus(dst + off, hardBloomScan, shape);
    }

    // Allow nearest three lines to affect pixel.
    inline float3 Tri(float2 pos, constant Uniforms &u,
                      texture2d<float, access::sample> src, sampler sam) {
        float3 a = Horz3(pos, -1.0, u, src, sam);
        float3 b = Horz5(pos,  0.0, u, src, sam);
        float3 c = Horz3(pos,  1.0, u, src, sam);

        float wa = Scan(pos, -1.0, u.hardScan, u.sourceSize, u.shape);
        float wb = Scan(pos,  0.0, u.hardScan, u.sourceSize, u.shape);
        float wc = Scan(pos,  1.0, u.hardScan, u.sourceSize, u.shape);

        return a * wa + b * wb + c * wc;
    }

    // Small bloom.
    inline float3 Bloom(float2 pos, constant Uniforms &u,
                        texture2d<float, access::sample> src, sampler sam) {
        float3 a = Horz5(pos, -2.0, u, src, sam);
        float3 b = Horz7(pos, -1.0, u, src, sam);
        float3 c = Horz7(pos,  0.0, u, src, sam);
        float3 d = Horz7(pos,  1.0, u, src, sam);
        float3 e = Horz5(pos,  2.0, u, src, sam);

        float wa = BloomScan(pos, -2.0, u.hardBloomScan, u.sourceSize, u.shape);
        float wb = BloomScan(pos, -1.0, u.hardBloomScan, u.sourceSize, u.shape);
        float wc = BloomScan(pos,  0.0, u.hardBloomScan, u.sourceSize, u.shape);
        float wd = BloomScan(pos,  1.0, u.hardBloomScan, u.sourceSize, u.shape);
        float we = BloomScan(pos,  2.0, u.hardBloomScan, u.sourceSize, u.shape);

        return a * wa + b * wb + c * wc + d * wd + e * we;
    }

    // Distortion of scanlines, and end of screen alpha.
    inline float2 Warp(float2 pos, float warpX, float warpY) {
        pos = pos * 2.0 - 1.0;
        pos *= float2(1.0 + (pos.y * pos.y) * warpX,
                       1.0 + (pos.x * pos.x) * warpY);
        return pos * 0.5 + 0.5;
    }

    // Shadow mask.
    inline float3 Mask(float2 pos, float maskDark, float maskLight, float shadowMask) {
        float3 mask = float3(maskDark);

        // Very compressed TV style shadow mask.
        if (shadowMask == 1.0) {
            float line = maskLight;
            float odd = 0.0;

            if (fract(pos.x * 0.166666666) < 0.5) odd = 1.0;
            if (fract((pos.y + odd) * 0.5) < 0.5) line = maskDark;

            pos.x = fract(pos.x * 0.333333333);

            if      (pos.x < 0.333) mask.r = maskLight;
            else if (pos.x < 0.666) mask.g = maskLight;
            else                    mask.b = maskLight;
            mask *= line;
        }

        // Aperture-grille.
        else if (shadowMask == 2.0) {
            pos.x = fract(pos.x * 0.333333333);

            if      (pos.x < 0.333) mask.r = maskLight;
            else if (pos.x < 0.666) mask.g = maskLight;
            else                    mask.b = maskLight;
        }

        // Stretched VGA style shadow mask (same as prior shaders).
        else if (shadowMask == 3.0) {
            pos.x += pos.y * 3.0;
            pos.x  = fract(pos.x * 0.166666666);

            if      (pos.x < 0.333) mask.r = maskLight;
            else if (pos.x < 0.666) mask.g = maskLight;
            else                    mask.b = maskLight;
        }

        // VGA style shadow mask.
        else if (shadowMask == 4.0) {
            pos.xy = floor(pos.xy * float2(1.0, 0.5));
            pos.x += pos.y * 3.0;
            pos.x  = fract(pos.x * 0.166666666);

            if      (pos.x < 0.333) mask.r = maskLight;
            else if (pos.x < 0.666) mask.g = maskLight;
            else                    mask.b = maskLight;
        }

        return mask;
    }

    // Compute kernel
    kernel void crtLottes(texture2d<float, access::sample> inTexture  [[ texture(0) ]],
                          texture2d<float, access::write>  outTexture [[ texture(1) ]],
                          constant Uniforms                &u         [[ buffer(0) ]],
                          sampler                          sam        [[ sampler(0) ]],
                          uint2                            gid        [[ thread_position_in_grid ]])
    {
        if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

        float2 outSize = float2(outTexture.get_width(), outTexture.get_height());
        float2 uvOut = (float2(gid) + 0.5) / outSize;

        // Apply warp distortion
        float2 pos = Warp(uvOut, u.warpX, u.warpY);

        // Main CRT color
        float3 outColor = Tri(pos, u, inTexture, sam);

        // Add bloom
        outColor += Bloom(pos, u, inTexture, sam) * u.bloomAmount;

        // Apply shadow mask
        if (u.shadowMask > 0.0) {
            // OutputSize.zw in slang is 1/OutputSize.xy (texel size)
            outColor *= Mask(uvOut * outSize * 1.000001, u.maskDark, u.maskLight, u.shadowMask);
        }

        outColor = ToSrgb(outColor, u.scaleInLinearGamma);

        outTexture.write(float4(outColor, 1.0), gid);
    }
}
