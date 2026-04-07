#include <metal_stdlib>
using namespace metal;

namespace vhs {

    struct Uniforms {
        float wiggle;
        float smear;
        int frameCount;
        int frameDirection;
        float2 resolution;
        float2 window;
    };

    // YIQ / RGB Conversions
    inline float3 rgb2yiq(float3 c) {
        return float3(
            (0.2989 * c.x + 0.5959 * c.y + 0.2115 * c.z),
            (0.5870 * c.x - 0.2744 * c.y - 0.5229 * c.z),
            (0.1140 * c.x - 0.3216 * c.y + 0.3114 * c.z)
        );
    }

    inline float3 yiq2rgb(float3 c) {
        return float3(
            (1.0 * c.x + 1.0 * c.y + 1.0 * c.z),
            (0.956 * c.x - 0.2720 * c.y - 1.1060 * c.z),
            (0.6210 * c.x - 0.6474 * c.y + 1.7046 * c.z)
        );
    }

    inline float2 Circle(float Start, float Points, float Point) {
        float Rad = (3.141592 * 2.0 * (1.0 / Points)) * (Point + Start);
        return float2(-(.3 + Rad), cos(Rad));
    }

    inline float3 Blur(float2 uv, float d, float iTime, texture2d<float, access::sample> tex, sampler s) {
        float b = 1.0;
        float t = 0.0;
        float2 PixelOffset = float2(d + .0005 * t, 0.0);

        float Start = 2.0 / 14.0;
        float2 Scale = 0.66 * 4.0 * 2.0 * PixelOffset;

        float3 N0 = tex.sample(s, uv + Circle(Start, 14.0, 0.0) * Scale).rgb;
        float3 N1 = tex.sample(s, uv + Circle(Start, 14.0, 1.0) * Scale).rgb;
        float3 N2 = tex.sample(s, uv + Circle(Start, 14.0, 2.0) * Scale).rgb;
        float3 N3 = tex.sample(s, uv + Circle(Start, 14.0, 3.0) * Scale).rgb;
        float3 N4 = tex.sample(s, uv + Circle(Start, 14.0, 4.0) * Scale).rgb;
        float3 N5 = tex.sample(s, uv + Circle(Start, 14.0, 5.0) * Scale).rgb;
        float3 N6 = tex.sample(s, uv + Circle(Start, 14.0, 6.0) * Scale).rgb;
        float3 N7 = tex.sample(s, uv + Circle(Start, 14.0, 7.0) * Scale).rgb;
        float3 N8 = tex.sample(s, uv + Circle(Start, 14.0, 8.0) * Scale).rgb;
        float3 N9 = tex.sample(s, uv + Circle(Start, 14.0, 9.0) * Scale).rgb;
        float3 N10 = tex.sample(s, uv + Circle(Start, 14.0, 10.0) * Scale).rgb;
        float3 N11 = tex.sample(s, uv + Circle(Start, 14.0, 11.0) * Scale).rgb;
        float3 N12 = tex.sample(s, uv + Circle(Start, 14.0, 12.0) * Scale).rgb;
        float3 N13 = tex.sample(s, uv + Circle(Start, 14.0, 13.0) * Scale).rgb;
        float3 N14 = tex.sample(s, uv).rgb;

        float4 clr = tex.sample(s, uv);
        float W = 1.0 / 15.0;

        clr.rgb = (N0 * W) + (N1 * W) + (N2 * W) + (N3 * W) + (N4 * W) +
                  (N5 * W) + (N6 * W) + (N7 * W) + (N8 * W) + (N9 * W) +
                  (N10 * W) + (N11 * W) + (N12 * W) + (N13 * W) + (N14 * W);
                  
        return clr.rgb * b;
    }

    inline float onOff(float a, float b, float c, float framecount) {
        return step(c, sin((framecount * 0.001) + a * cos((framecount * 0.001) * b)));
    }

    inline float2 jumpy(float2 uv, float framecount, float wiggle) {
        float2 look = uv;
        float window = 1.0 / (1.0 + 80.0 * (look.y - fmod(framecount / 4.0, 1.0)) * (look.y - fmod(framecount / 4.0, 1.0)));
        look.x += 0.05 * sin(look.y * 10.0 + framecount) / 20.0 * onOff(4.0, 4.0, 0.3, framecount) * (0.5 + cos(framecount * 20.0)) * window;
        float vShift = (0.1 * wiggle) * 0.4 * onOff(2.0, 3.0, 0.9, framecount) * (sin(framecount) * sin(framecount * 20.0) + (0.5 + 0.1 * sin(framecount * 200.0) * cos(framecount)));
        look.y = fmod(look.y - 0.01 * vShift, 1.0);
        
        // Ensure wrap behavior for fmod
        if (look.y < 0.0) look.y += 1.0;
        
        return look;
    }

    kernel void vhsEffect(texture2d<float, access::sample> inTexture [[texture(0)]],
                          texture2d<float, access::write> outTexture [[texture(1)]],
                          constant Uniforms &uniforms [[buffer(0)]],
                          sampler linearSampler [[sampler(0)]],
                          uint2 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) {
            return;
        }

        float2 texSize = float2(inTexture.get_width(), inTexture.get_height());
        float2 vTexCoord = float2(gid) / texSize;

        float iTime = fmod(float(uniforms.frameCount), 7.0);

        float d = 0.1 - ceil(fmod(iTime / 3.0, 1.0) + 0.5) * 0.1;
        float2 uv = jumpy(vTexCoord, iTime, uniforms.wiggle);

        float s = 0.0001 * -d + 0.0001 * uniforms.wiggle * sin(iTime);
        float e = min(.30, pow(max(0.0, cos(uv.y * 4.0 + .3) - .75) * (s + 0.5) * 1.0, 3.0)) * 25.0;
        float r = (iTime * (2.0 * s));
        
        uv.x += abs(r * pow(min(.003, (-uv.y + (.01 * fmod(iTime, 17.0)))) * 3.0, 2.0));

        d = .051 + abs(sin(s / 4.0));
        float c = max(0.0001, .002 * d) * uniforms.smear;
        
        float4 finalColor;
        
        // Y Channel Blur
        finalColor.xyz = Blur(uv, c + c * uv.x, iTime, inTexture, linearSampler);
        float y = rgb2yiq(finalColor.xyz).r;

        // I Channel Blur
        uv.x += .01 * d;
        c *= 6.0;
        finalColor.xyz = Blur(uv, c, iTime, inTexture, linearSampler);
        float i = rgb2yiq(finalColor.xyz).g;

        // Q Channel Blur
        uv.x += .005 * d;
        c *= 2.50;
        finalColor.xyz = Blur(uv, c, iTime, inTexture, linearSampler);
        float q = rgb2yiq(finalColor.xyz).b;

        finalColor = float4(yiq2rgb(float3(y, i, q)) - pow(s + e * 2.0, 3.0), 1.0);

        outTexture.write(finalColor, gid);
    }
}
