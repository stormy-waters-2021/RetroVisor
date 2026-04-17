// -----------------------------------------------------------------------------
// This file is part of RetroVisor
//
// Copyright (C) Dirk W. Hoffmann. www.dirkwhoffmann.de
// Licensed under the GNU General Public License v3
//
// See https://www.gnu.org for license information
// -----------------------------------------------------------------------------

import MetalKit

final class ResIndependentScanlines: Shader {

    struct Uniforms {

        var sourceSize: SIMD2<Float>
        var outputSize: SIMD2<Float>

        var amp: Float
        var phase: Float
        var lines_black: Float
        var lines_white: Float
        var mask: Float
        var mask_weight: Float
        var fauxRes: Float
        var autoscale: Float

        static let defaults = Uniforms(
            sourceSize: [0, 0],
            outputSize: [0, 0],
            amp: 1.25,
            phase: 0.5,
            lines_black: 0.0,
            lines_white: 1.0,
            mask: 0.0,
            mask_weight: 0.5,
            fauxRes: 224.0,
            autoscale: 0.0
        )
    }

    var kernel: Kernel!
    var uniforms: Uniforms = .defaults

    init() {

        super.init(name: "ResIndependentScanlines")

        settings = [

            Group(title: "Scanlines", [

                ShaderSetting(
                    title: "Amplitude",
                    range: 0.0...2.0, step: 0.05,
                    value: Binding(
                        key: "amp",
                        get: { [unowned self] in self.uniforms.amp },
                        set: { [unowned self] in self.uniforms.amp = $0 })
                ),

                ShaderSetting(
                    title: "Phase",
                    range: 0.0...2.0, step: 0.05,
                    value: Binding(
                        key: "phase",
                        get: { [unowned self] in self.uniforms.phase },
                        set: { [unowned self] in self.uniforms.phase = $0 })
                ),

                ShaderSetting(
                    title: "Lines Blacks",
                    range: 0.0...1.0, step: 0.05,
                    value: Binding(
                        key: "lines_black",
                        get: { [unowned self] in self.uniforms.lines_black },
                        set: { [unowned self] in self.uniforms.lines_black = $0 })
                ),

                ShaderSetting(
                    title: "Lines Whites",
                    range: 0.0...2.0, step: 0.05,
                    value: Binding(
                        key: "lines_white",
                        get: { [unowned self] in self.uniforms.lines_white },
                        set: { [unowned self] in self.uniforms.lines_white = $0 })
                ),
            ]),

            Group(title: "Mask", [

                ShaderSetting(
                    title: "Mask Layout",
                    range: 0.0...19.0, step: 1.0,
                    value: Binding(
                        key: "mask",
                        get: { [unowned self] in self.uniforms.mask },
                        set: { [unowned self] in self.uniforms.mask = $0 })
                ),

                ShaderSetting(
                    title: "Mask Weight",
                    range: 0.0...1.0, step: 0.01,
                    value: Binding(
                        key: "mask_weight",
                        get: { [unowned self] in self.uniforms.mask_weight },
                        set: { [unowned self] in self.uniforms.mask_weight = $0 })
                ),
            ]),

            Group(title: "Scale", [

                ShaderSetting(
                    title: "Simulated Image Height",
                    range: 144.0...1080.0, step: 1.0,
                    value: Binding(
                        key: "fauxRes",
                        get: { [unowned self] in self.uniforms.fauxRes },
                        set: { [unowned self] in self.uniforms.fauxRes = $0 })
                ),

                ShaderSetting(
                    title: "Automatic Scale",
                    range: nil, step: 1.0,
                    value: Binding(
                        key: "autoscale",
                        get: { [unowned self] in self.uniforms.autoscale },
                        set: { [unowned self] in self.uniforms.autoscale = $0 })
                ),
            ])
        ]
    }

    override func revertToPreset(nr: Int) {

        uniforms = Uniforms.defaults
    }

    override func activate() {

        super.activate()
        kernel = ScanlineKernel(sampler: ShaderLibrary.linear)
    }

    override func apply(commandBuffer: MTLCommandBuffer,
                        in input: MTLTexture, out output: MTLTexture, rect: CGRect) {

        let metalView = app.windowController!.metalView!
        uniforms.sourceSize = metalView.uniforms.resolution
        uniforms.outputSize = metalView.uniforms.window

        kernel.apply(commandBuffer: commandBuffer,
                     source: input, target: output,
                     options: &uniforms,
                     length: MemoryLayout<Uniforms>.stride)
    }
}

extension ResIndependentScanlines {

    class ScanlineKernel: Kernel {

        convenience init?(sampler: MTLSamplerState) {

            self.init(name: "resindepscan::resIndependentScanlines", sampler: sampler)
        }
    }
}
