// -----------------------------------------------------------------------------
// This file is part of RetroVisor
//
// MSL port of crt-lottes by Timothy Lottes (public domain)
// Ported from: https://github.com/libretro/slang-shaders/blob/master/crt/shaders/crt-lottes.slang
// -----------------------------------------------------------------------------

import MetalKit

final class CRTLottes: Shader {

    struct Uniforms {

        var hardScan: Float
        var hardPix: Float
        var warpX: Float
        var warpY: Float
        var maskDark: Float
        var maskLight: Float
        var scaleInLinearGamma: Float
        var shadowMask: Float
        var brightBoost: Float
        var hardBloomScan: Float
        var hardBloomPix: Float
        var bloomAmount: Float
        var shape: Float

        var sourceSize: SIMD2<Float>
        var outputSize: SIMD2<Float>

        static let defaults = Uniforms(
            hardScan: -8.0,
            hardPix: -3.0,
            warpX: 0.031,
            warpY: 0.041,
            maskDark: 0.5,
            maskLight: 1.5,
            scaleInLinearGamma: 1.0,
            shadowMask: 3.0,
            brightBoost: 1.0,
            hardBloomScan: -2.0,
            hardBloomPix: -1.5,
            bloomAmount: 0.15,
            shape: 2.0,

            sourceSize: [0, 0],
            outputSize: [0, 0]
        )
    }

    var kernel: Kernel!
    var uniforms: Uniforms = .defaults

    init() {

        super.init(name: "CRT Lottes")

        settings = [

            Group(title: "Scanline", [

                ShaderSetting(
                    title: "Scanline Hardness",
                    range: -20.0...0.0, step: 1.0,
                    value: Binding(
                        key: "hardScan",
                        get: { [unowned self] in self.uniforms.hardScan },
                        set: { [unowned self] in self.uniforms.hardScan = $0 })
                ),

                ShaderSetting(
                    title: "Pixel Hardness",
                    range: -20.0...0.0, step: 1.0,
                    value: Binding(
                        key: "hardPix",
                        get: { [unowned self] in self.uniforms.hardPix },
                        set: { [unowned self] in self.uniforms.hardPix = $0 })
                ),

                ShaderSetting(
                    title: "Filter Kernel Shape",
                    range: 0.0...10.0, step: 0.05,
                    value: Binding(
                        key: "shape",
                        get: { [unowned self] in self.uniforms.shape },
                        set: { [unowned self] in self.uniforms.shape = $0 })
                ),
            ]),

            Group(title: "Warp", [

                ShaderSetting(
                    title: "Warp X",
                    range: 0.0...0.125, step: 0.01,
                    value: Binding(
                        key: "warpX",
                        get: { [unowned self] in self.uniforms.warpX },
                        set: { [unowned self] in self.uniforms.warpX = $0 })
                ),

                ShaderSetting(
                    title: "Warp Y",
                    range: 0.0...0.125, step: 0.01,
                    value: Binding(
                        key: "warpY",
                        get: { [unowned self] in self.uniforms.warpY },
                        set: { [unowned self] in self.uniforms.warpY = $0 })
                ),
            ]),

            Group(title: "Shadow Mask", [

                ShaderSetting(
                    title: "Shadow Mask Type",
                    range: 0.0...4.0, step: 1.0,
                    value: Binding(
                        key: "shadowMask",
                        get: { [unowned self] in self.uniforms.shadowMask },
                        set: { [unowned self] in self.uniforms.shadowMask = $0 })
                ),

                ShaderSetting(
                    title: "Mask Dark",
                    range: 0.0...2.0, step: 0.1,
                    value: Binding(
                        key: "maskDark",
                        get: { [unowned self] in self.uniforms.maskDark },
                        set: { [unowned self] in self.uniforms.maskDark = $0 })
                ),

                ShaderSetting(
                    title: "Mask Light",
                    range: 0.0...2.0, step: 0.1,
                    value: Binding(
                        key: "maskLight",
                        get: { [unowned self] in self.uniforms.maskLight },
                        set: { [unowned self] in self.uniforms.maskLight = $0 })
                ),
            ]),

            Group(title: "Bloom", [

                ShaderSetting(
                    title: "Bloom Amount",
                    range: 0.0...1.0, step: 0.05,
                    value: Binding(
                        key: "bloomAmount",
                        get: { [unowned self] in self.uniforms.bloomAmount },
                        set: { [unowned self] in self.uniforms.bloomAmount = $0 })
                ),

                ShaderSetting(
                    title: "Bloom X Softness",
                    range: -2.0...(-0.5), step: 0.1,
                    value: Binding(
                        key: "hardBloomPix",
                        get: { [unowned self] in self.uniforms.hardBloomPix },
                        set: { [unowned self] in self.uniforms.hardBloomPix = $0 })
                ),

                ShaderSetting(
                    title: "Bloom Y Softness",
                    range: -4.0...(-1.0), step: 0.1,
                    value: Binding(
                        key: "hardBloomScan",
                        get: { [unowned self] in self.uniforms.hardBloomScan },
                        set: { [unowned self] in self.uniforms.hardBloomScan = $0 })
                ),
            ]),

            Group(title: "Color", [

                ShaderSetting(
                    title: "Brightness Boost",
                    range: 0.0...2.0, step: 0.05,
                    value: Binding(
                        key: "brightBoost",
                        get: { [unowned self] in self.uniforms.brightBoost },
                        set: { [unowned self] in self.uniforms.brightBoost = $0 })
                ),

                ShaderSetting(
                    title: "Scale In Linear Gamma",
                    range: 0.0...1.0, step: 1.0,
                    value: Binding(
                        key: "scaleInLinearGamma",
                        get: { [unowned self] in self.uniforms.scaleInLinearGamma },
                        set: { [unowned self] in self.uniforms.scaleInLinearGamma = $0 })
                ),
            ]),
        ]
    }

    override func revertToPreset(nr: Int) {

        uniforms = Uniforms.defaults
    }

    override func activate() {

        super.activate()
        kernel = CrtLottesKernel(sampler: ShaderLibrary.nearest)
    }

    override func apply(commandBuffer: MTLCommandBuffer,
                        in input: MTLTexture, out output: MTLTexture, rect: CGRect) {

        uniforms.sourceSize = app.windowController!.metalView!.uniforms.resolution
        uniforms.outputSize = app.windowController!.metalView!.uniforms.window

        kernel.apply(commandBuffer: commandBuffer,
                     source: input, target: output,
                     options: &uniforms,
                     length: MemoryLayout<Uniforms>.stride)
    }
}

extension CRTLottes {

    class CrtLottesKernel: Kernel {

        convenience init?(sampler: MTLSamplerState) {

            self.init(name: "crtlottes::crtLottes", sampler: sampler)
        }
    }
}
