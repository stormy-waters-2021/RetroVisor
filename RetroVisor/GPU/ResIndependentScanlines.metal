// -----------------------------------------------------------------------------
// This file is part of RetroVisor
//
// Copyright (C) Dirk W. Hoffmann. www.dirkwhoffmann.de
// Licensed under the GNU General Public License v3
//
// See https://www.gnu.org for license information
// -----------------------------------------------------------------------------

// MSL port of res-independent-scanlines by RiskyJumps (public domain)
// Subpixel mask collection by hunterk (public domain)

#include <metal_stdlib>

using namespace metal;

namespace resindepscan {

    struct Uniforms {

        float2 sourceSize;   // Original source resolution
        float2 outputSize;   // Output/window resolution

        float amp;           // Amplitude
        float phase;         // Phase
        float lines_black;   // Lines Blacks
        float lines_white;   // Lines Whites
        float mask;          // Mask Layout (0-19)
        float mask_weight;   // Mask Weight
        float fauxRes;       // Simulated Image Height
        float autoscale;     // Automatic Scale
    };

    constant constexpr float freq   = 0.5;
    constant constexpr float offset = 0.0;
    constant constexpr float pi     = 3.141592654;

    // -------------------------------------------------------------------
    // Subpixel mask weights (adapted from hunterk's subpixel_masks.h)
    // -------------------------------------------------------------------
    inline float3 mask_weights(float2 coord, float mask_intensity, int phosphor_layout) {

        float3 weights = float3(1.0);
        float on  = 1.0;
        float off = 1.0 - mask_intensity;

        float3 red     = float3(on,  off, off);
        float3 green   = float3(off, on,  off);
        float3 blue    = float3(off, off, on );
        float3 magenta = float3(on,  off, on );
        float3 yellow  = float3(on,  on,  off);
        float3 cyan    = float3(off, on,  on );
        float3 black   = float3(off, off, off);
        float3 white   = float3(on,  on,  on );

        int w = 0, z = 0;

        float3 aperture_weights = mix(magenta, green, floor(fmod(coord.x, 2.0)));

        if (phosphor_layout == 0) {
            return weights;
        }
        else if (phosphor_layout == 1) {
            // classic aperture for RGB panels
            return aperture_weights;
        }
        else if (phosphor_layout == 2) {
            // 2x2 shadow mask for RGB panels
            float3 inverse_aperture = mix(green, magenta, floor(fmod(coord.x, 2.0)));
            return mix(aperture_weights, inverse_aperture, floor(fmod(coord.y, 2.0)));
        }
        else if (phosphor_layout == 3) {
            // slot mask for RGB panels
            float3 slotmask[3][4] = {
                {magenta, green, black,   black},
                {magenta, green, magenta, green},
                {black,   black, magenta, green}
            };
            w = int(floor(fmod(coord.y, 3.0)));
            z = int(floor(fmod(coord.x, 4.0)));
            return slotmask[w][z];
        }
        else if (phosphor_layout == 4) {
            // classic aperture for RBG panels
            return mix(yellow, blue, floor(fmod(coord.x, 2.0)));
        }
        else if (phosphor_layout == 5) {
            // 2x2 shadow mask for RBG panels
            float3 inverse_aperture = mix(blue, yellow, floor(fmod(coord.x, 2.0)));
            return mix(mix(yellow, blue, floor(fmod(coord.x, 2.0))), inverse_aperture, floor(fmod(coord.y, 2.0)));
        }
        else if (phosphor_layout == 6) {
            // aperture_1_4_rgb
            float3 ap4[4] = {red, green, blue, black};
            z = int(floor(fmod(coord.x, 4.0)));
            return ap4[z];
        }
        else if (phosphor_layout == 7) {
            // aperture_2_5_bgr
            float3 ap3[5] = {red, magenta, blue, green, green};
            z = int(floor(fmod(coord.x, 5.0)));
            return ap3[z];
        }
        else if (phosphor_layout == 8) {
            // aperture_3_6_rgb
            float3 big_ap[7] = {red, red, yellow, green, cyan, blue, blue};
            w = int(floor(fmod(coord.x, 7.0)));
            return big_ap[w];
        }
        else if (phosphor_layout == 9) {
            // reduced TVL aperture for RGB panels
            float3 big_ap_rgb[4] = {red, yellow, cyan, blue};
            w = int(floor(fmod(coord.x, 4.0)));
            return big_ap_rgb[w];
        }
        else if (phosphor_layout == 10) {
            // reduced TVL aperture for RBG panels
            float3 big_ap_rbg[4] = {red, magenta, cyan, green};
            w = int(floor(fmod(coord.x, 4.0)));
            return big_ap_rbg[w];
        }
        else if (phosphor_layout == 11) {
            // delta_1_4x1_rgb
            float3 delta1[2][4] = {
                {red,  green, blue, black},
                {blue, black, red,  green}
            };
            w = int(floor(fmod(coord.y, 2.0)));
            z = int(floor(fmod(coord.x, 4.0)));
            return delta1[w][z];
        }
        else if (phosphor_layout == 12) {
            // delta_2_4x1_rgb
            float3 delta[2][4] = {
                {red, yellow, cyan, blue},
                {cyan, blue, red, yellow}
            };
            w = int(floor(fmod(coord.y, 2.0)));
            z = int(floor(fmod(coord.x, 4.0)));
            return delta[w][z];
        }
        else if (phosphor_layout == 13) {
            // delta_2_4x2_rgb
            float3 delta[4][4] = {
                {red,  yellow, cyan, blue},
                {red,  yellow, cyan, blue},
                {cyan, blue,   red,  yellow},
                {cyan, blue,   red,  yellow}
            };
            w = int(floor(fmod(coord.y, 4.0)));
            z = int(floor(fmod(coord.x, 4.0)));
            return delta[w][z];
        }
        else if (phosphor_layout == 14) {
            // slot mask for RGB panels; low-pitch
            float3 slotmask[3][6] = {
                {magenta, green, black, black,   black, black},
                {magenta, green, black, magenta, green, black},
                {black,   black, black, magenta, green, black}
            };
            w = int(floor(fmod(coord.y, 3.0)));
            z = int(floor(fmod(coord.x, 6.0)));
            return slotmask[w][z];
        }
        else if (phosphor_layout == 15) {
            // slot_2_4x4_rgb
            float3 slot2[4][8] = {
                {red,   yellow, cyan,  blue,  red,   yellow, cyan,  blue },
                {red,   yellow, cyan,  blue,  black, black,  black, black},
                {red,   yellow, cyan,  blue,  red,   yellow, cyan,  blue },
                {black, black,  black, black, red,   yellow, cyan,  blue }
            };
            w = int(floor(fmod(coord.y, 4.0)));
            z = int(floor(fmod(coord.x, 8.0)));
            return slot2[w][z];
        }
        else if (phosphor_layout == 16) {
            // slot mask for RBG panels
            float3 slotmask[3][4] = {
                {yellow, blue,  black,  black},
                {yellow, blue,  yellow, blue},
                {black,  black, yellow, blue}
            };
            w = int(floor(fmod(coord.y, 3.0)));
            z = int(floor(fmod(coord.x, 4.0)));
            return slotmask[w][z];
        }
        else if (phosphor_layout == 17) {
            // slot_2_5x4_bgr
            float3 slot2[4][10] = {
                {red,   magenta, blue,  green, green, red,   magenta, blue,  green, green},
                {black, blue,    blue,  green, green, red,   red,     black, black, black},
                {red,   magenta, blue,  green, green, red,   magenta, blue,  green, green},
                {red,   red,     black, black, black, black, blue,    blue,  green, green}
            };
            w = int(floor(fmod(coord.y, 4.0)));
            z = int(floor(fmod(coord.x, 10.0)));
            return slot2[w][z];
        }
        else if (phosphor_layout == 18) {
            // same as above but for RBG panels
            float3 slot2[4][10] = {
                {red,   yellow, green, blue,  blue,  red,   yellow, green, blue,  blue },
                {black, green,  green, blue,  blue,  red,   red,    black, black, black},
                {red,   yellow, green, blue,  blue,  red,   yellow, green, blue,  blue },
                {red,   red,    black, black, black, black, green,  green, blue,  blue }
            };
            w = int(floor(fmod(coord.y, 4.0)));
            z = int(floor(fmod(coord.x, 10.0)));
            return slot2[w][z];
        }
        else if (phosphor_layout == 19) {
            // slot_3_7x6_rgb
            float3 slot[6][14] = {
                {red,   red,   yellow, green, cyan,  blue,  blue,  red,   red,   yellow, green,  cyan,  blue,  blue},
                {red,   red,   yellow, green, cyan,  blue,  blue,  red,   red,   yellow, green,  cyan,  blue,  blue},
                {red,   red,   yellow, green, cyan,  blue,  blue,  black, black, black,  black,  black, black, black},
                {red,   red,   yellow, green, cyan,  blue,  blue,  red,   red,   yellow, green,  cyan,  blue,  blue},
                {red,   red,   yellow, green, cyan,  blue,  blue,  red,   red,   yellow, green,  cyan,  blue,  blue},
                {black, black, black,  black, black, black, black, black, red,   red,    yellow, green, cyan,  blue}
            };
            w = int(floor(fmod(coord.y, 6.0)));
            z = int(floor(fmod(coord.x, 14.0)));
            return slot[w][z];
        }

        return weights;
    }

    // -------------------------------------------------------------------
    // Main compute kernel
    // -------------------------------------------------------------------
    kernel void resIndependentScanlines(
        texture2d<float, access::sample> source   [[texture(0)]],
        texture2d<float, access::write>  target   [[texture(1)]],
        constant Uniforms               &uniforms [[buffer(0)]],
        sampler                          sam       [[sampler(0)]],
        uint2                            gid       [[thread_position_in_grid]])
    {
        uint tw = target.get_width();
        uint th = target.get_height();
        if (gid.x >= tw || gid.y >= th) return;

        float2 texSize = float2(tw, th);
        float2 uv = (float2(gid) + 0.5) / texSize;

        // Sample the source texture
        float3 color = source.sample(sam, uv).rgb;

        // Compute scanline
        float scale = uniforms.fauxRes;
        if (uniforms.autoscale == 1.0) scale = uniforms.sourceSize.y;

        float omega = 2.0 * pi * freq;
        float angle = (float(gid.y) / texSize.y) * omega * scale + uniforms.phase;

        float lines = sin(angle);
        lines *= uniforms.amp;
        lines += offset;
        lines = fabs(lines);
        lines *= uniforms.lines_white - uniforms.lines_black;
        lines += uniforms.lines_black;
        color *= lines;

        // Apply subpixel mask
        color *= mask_weights(float2(gid), uniforms.mask_weight, int(uniforms.mask));

        target.write(float4(color, 1.0), gid);
    }
}
