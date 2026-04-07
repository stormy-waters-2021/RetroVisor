// -----------------------------------------------------------------------------
// This file is part of RetroVisor
//
// Copyright (C) Dirk W. Hoffmann. www.dirkwhoffmann.de
// Licensed under the GNU General Public License v3
//
// See https://www.gnu.org for license information
// -----------------------------------------------------------------------------

import MetalKit

final class VHS: Shader {

    struct Uniforms {

        var wiggle: Float
        var smear: Float
        var frameCount: Int32
        var frameDirection: Int32
        
        var resolution: SIMD2<Float>
        var window: SIMD2<Float>

        static let defaults = Uniforms(

            wiggle: 0.0,
            smear: 0.5,
            frameCount: 0,
            frameDirection: 1,
            
            resolution: [0,0],
            window: [0,0]
        )
    }
    
    var kernel: Kernel!
    var uniforms: Uniforms = .defaults

    // Input texture passed to the VHS kernel
    var src: MTLTexture!

    init() {

        super.init(name: "VHS")

        settings = [

            Group(title: "Uniforms", [
                
                ShaderSetting(
                    title: "Wiggle",
                    range: 0.0...10.0, step: 0.01,
                    value: Binding(
                        key: "wiggle",
                        get: { [unowned self] in self.uniforms.wiggle },
                        set: { [unowned self] in self.uniforms.wiggle = $0 }),
                ),
                
                ShaderSetting(
                    title: "Chroma Smear",
                    range: 0.0...1.0, step: 0.05,
                    value: Binding(
                        key: "smear",
                        get: { [unowned self] in self.uniforms.smear },
                        set: { [unowned self] in self.uniforms.smear = $0 }),
                )
            ])
        ]
    }

    override func revertToPreset(nr: Int) {
        
        uniforms = Uniforms.defaults
    }

    override func activate() {

        super.activate()
        // Make sure to match the namespace/function name in your .metal file
        kernel = VHSKernel(sampler: ShaderLibrary.linear)
    }

    func updateTextures(in input: MTLTexture, out output: MTLTexture, rect: CGRect) {

        let inpWidth = output.width
        let inpHeight = output.height

        if src?.width != inpWidth || src?.height != inpHeight {

            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: output.pixelFormat,
                width: inpWidth,
                height: inpHeight,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite, .renderTarget]
            src = output.device.makeTexture(descriptor: desc)
        }
    }
    
    override func apply(commandBuffer: MTLCommandBuffer,
                        in input: MTLTexture, out output: MTLTexture, rect: CGRect) {

        updateTextures(in: input, out: output, rect: rect)

        src = input
        
        // Update frame count (simulating iTime in the original shader)
        uniforms.frameCount &+= 1
        
        // Setup uniforms
        uniforms.resolution = app.windowController!.metalView!.uniforms.resolution
        uniforms.window = app.windowController!.metalView!.uniforms.window

        // Apply the VHS kernel
        kernel.apply(commandBuffer: commandBuffer,
                     source: src, target: output,
                     options: &uniforms,
                     length: MemoryLayout<Uniforms>.stride)
    }
}

extension VHS {
    
    class VHSKernel: Kernel {

        convenience init?(sampler: MTLSamplerState) {

            // Maps to the compute function in vhs.metal
            self.init(name: "vhs::vhsEffect", sampler: sampler)
        }
    }
}
