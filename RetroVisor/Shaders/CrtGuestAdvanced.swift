// -----------------------------------------------------------------------------
// This file is part of RetroVisor
//
// MSL port of crt-guest-advanced by guest(r) and Dr. Venom
// Original: https://github.com/libretro/slang-shaders/tree/master/crt/shaders/guest/advanced
// Licensed under the GNU General Public License v2+
// -----------------------------------------------------------------------------

import MetalKit
import MetalPerformanceShaders

@MainActor
final class CRTGuestAdvanced: Shader {

    // Must match the Metal Uniforms struct in CrtGuestAdvanced.metal
    struct Uniforms {

        var sourceSize: SIMD2<Float>
        var outputSize: SIMD2<Float>
        var frameCount: UInt32

        // Afterglow
        var PR: Float; var PG: Float; var PB: Float; var esrc: Float; var bth: Float

        // Pre-shaders / color
        var AS: Float; var agsat: Float; var CS: Float; var CP: Float; var TNTC: Float
        var LS: Float; var WP: Float; var wp_saturation: Float; var pre_bb: Float
        var contr: Float; var pre_gc: Float; var sega_fix: Float; var BP: Float
        var vigstr: Float; var vigdef: Float

        // Raster bloom / avg lum
        var lsmooth: Float; var lsdev: Float; var OS: Float; var BLOOM: Float

        // Gamma / interlace
        var GAMMA_INPUT: Float; var gamma_out: Float; var inter: Float; var interm: Float
        var iscan: Float; var intres: Float; var downsample_levelx: Float; var downsample_levely: Float
        var iscans: Float; var vga_mode: Float; var hiscan: Float

        // Magic glow
        var m_glow: Float; var m_glow_cutoff: Float; var m_glow_low: Float
        var m_glow_high: Float; var m_glow_dist: Float; var m_glow_mask: Float

        // Glow pass
        var FINE_GLOW: Float; var SIZEH: Float; var SIGMA_H: Float
        var SIZEV: Float; var SIGMA_V: Float

        // Bloom pass
        var FINE_BLOOM: Float; var SIZEHB: Float; var SIGMA_HB: Float
        var SIZEVB: Float; var SIGMA_VB: Float

        // Brightness
        var glow: Float; var bloom: Float; var mask_bloom: Float; var bloom_dist: Float
        var halation: Float; var bmask1: Float; var hmask1: Float; var gamma_c: Float
        var gamma_c2: Float; var brightboost: Float; var brightboost1: Float; var clips: Float

        // Scanline
        var gsl: Float; var scanline1: Float; var scanline2: Float
        var beam_min: Float; var beam_max: Float; var tds: Float; var beam_size: Float
        var scans: Float; var scan_falloff: Float; var spike: Float
        var scangamma: Float; var rolling_scan: Float; var no_scanlines: Float

        // Filtering
        var h_sharp: Float; var s_sharp: Float; var ring: Float
        var smart_ei: Float; var ei_limit: Float; var sth: Float

        // Screen
        var TATE: Float; var IOS: Float; var warpX: Float; var warpY: Float
        var c_shape: Float; var overscanX: Float; var overscanY: Float; var VShift: Float

        // Mask
        var shadowMask: Float; var maskstr: Float; var mcut: Float; var maskboost: Float
        var masksize: Float; var mask_zoom: Float; var mzoom_sh: Float; var mshift: Float
        var mask_layout: Float; var maskDark: Float; var maskLight: Float; var mask_gamma: Float
        var slotmask: Float; var slotmask1: Float; var slotwidth: Float; var double_slot: Float
        var slotms: Float; var smoothmask: Float; var smask_mit: Float; var bmask: Float
        var mclip: Float; var pr_scan: Float; var edgemask: Float

        // Border / corner
        var csize: Float; var bsize1: Float; var sborder: Float

        // Hum bar
        var barspeed: Float; var barintensity: Float; var bardir: Float

        // Deconvergence
        var dctypex: Float; var dctypey: Float
        var deconrr: Float; var deconrg: Float; var deconrb: Float
        var deconrry: Float; var deconrgy: Float; var deconrby: Float; var decons: Float

        // Post / noise
        var addnoised: Float; var noiseresd: Float; var noisetype: Float
        var post_br: Float; var oimage: Float

        static let defaults = Uniforms(
            sourceSize: [0, 0],
            outputSize: [0, 0],
            frameCount: 0,
            PR: 0.32, PG: 0.32, PB: 0.32, esrc: 1.0, bth: 4.0,
            AS: 0.20, agsat: 0.50, CS: 0, CP: 0, TNTC: 0,
            LS: 32, WP: 0, wp_saturation: 1.0, pre_bb: 1.0,
            contr: 0, pre_gc: 1.0, sega_fix: 0, BP: 0,
            vigstr: 0, vigdef: 1.0,
            lsmooth: 0.70, lsdev: 0.0, OS: 1, BLOOM: 0,
            GAMMA_INPUT: 2.4, gamma_out: 2.4, inter: 375, interm: 1,
            iscan: 0.20, intres: 0, downsample_levelx: 0, downsample_levely: 0,
            iscans: 0.25, vga_mode: 0, hiscan: 0,
            m_glow: 0, m_glow_cutoff: 0.12, m_glow_low: 0.35,
            m_glow_high: 5.0, m_glow_dist: 1.0, m_glow_mask: 1.0,
            FINE_GLOW: 1.0, SIZEH: 6, SIGMA_H: 1.20,
            SIZEV: 6, SIGMA_V: 1.20,
            FINE_BLOOM: 1.0, SIZEHB: 3, SIGMA_HB: 0.75,
            SIZEVB: 3, SIGMA_VB: 0.60,
            glow: 0.08, bloom: 0, mask_bloom: 0, bloom_dist: 0,
            halation: 0, bmask1: 0, hmask1: 0.35, gamma_c: 1.0,
            gamma_c2: 1.0, brightboost: 1.40, brightboost1: 1.10, clips: 0,
            gsl: 0, scanline1: 6, scanline2: 8,
            beam_min: 1.30, beam_max: 1.00, tds: 0, beam_size: 0.60,
            scans: 0.50, scan_falloff: 1.0, spike: 1.0,
            scangamma: 2.40, rolling_scan: 0, no_scanlines: 0,
            h_sharp: 5.20, s_sharp: 0.50, ring: 0,
            smart_ei: 0, ei_limit: 0.25, sth: 0.23,
            TATE: 0, IOS: 0, warpX: 0, warpY: 0,
            c_shape: 0.25, overscanX: 0, overscanY: 0, VShift: 0,
            shadowMask: 0, maskstr: 0.3, mcut: 1.10, maskboost: 1.0,
            masksize: 1, mask_zoom: 0, mzoom_sh: 0, mshift: 0,
            mask_layout: 0, maskDark: 0.5, maskLight: 1.5, mask_gamma: 2.40,
            slotmask: 0, slotmask1: 0, slotwidth: 0, double_slot: 2,
            slotms: 1, smoothmask: 0, smask_mit: 0, bmask: 0,
            mclip: 0, pr_scan: 0.10, edgemask: 0,
            csize: 0, bsize1: 0, sborder: 0.75,
            barspeed: 50, barintensity: 0, bardir: 0,
            dctypex: 0, dctypey: 0,
            deconrr: 0, deconrg: 0, deconrb: 0,
            deconrry: 0, deconrgy: 0, deconrby: 0, decons: 1.0,
            addnoised: 0, noiseresd: 2, noisetype: 0,
            post_br: 1.0, oimage: 0
        )
    }

    var uniforms: Uniforms = .defaults
    private var frameCounter: UInt32 = 0

    // Intermediate textures
    var afterglowTexA: MTLTexture!  // feedback ping
    var afterglowTexB: MTLTexture!  // feedback pong
    var prePassTex: MTLTexture!
    var avgLumTexA: MTLTexture!     // feedback ping
    var avgLumTexB: MTLTexture!     // feedback pong
    var linearizeTex: MTLTexture!
    var gaussHTex: MTLTexture!
    var glowTex: MTLTexture!
    var bloomHTex: MTLTexture!
    var bloomTex: MTLTexture!
    var crtTex: MTLTexture!

    // LUT textures
    var lut1: MTLTexture!
    var lut2: MTLTexture!
    var lut3: MTLTexture!
    var lut4: MTLTexture!

    // Feedback frame toggle
    var feedbackFrame: Bool = false

    // Compute kernels
    var afterglowKernel: Kernel!
    var preShadersKernel: Kernel!
    var avgLumKernel: Kernel!
    var linearizeKernel: Kernel!
    var gaussHKernel: Kernel!
    var gaussVKernel: Kernel!
    var bloomHKernel: Kernel!
    var bloomVKernel: Kernel!
    var crtKernel: Kernel!
    var deconvKernel: Kernel!

    init() {

        super.init(name: "CRT Guest Advanced")

        delegate = self

        settings = [

            Group(title: "Afterglow", [
                floatSetting("Persistence Red", key: "PR", range: 0...1, step: 0.01, get: { self.uniforms.PR }, set: { self.uniforms.PR = $0 }),
                floatSetting("Persistence Green", key: "PG", range: 0...1, step: 0.01, get: { self.uniforms.PG }, set: { self.uniforms.PG = $0 }),
                floatSetting("Persistence Blue", key: "PB", range: 0...1, step: 0.01, get: { self.uniforms.PB }, set: { self.uniforms.PB = $0 }),
                floatSetting("Afterglow Threshold", key: "bth", range: 0...255, step: 1.0, get: { self.uniforms.bth }, set: { self.uniforms.bth = $0 }),
            ]),

            Group(title: "Color Correction", [
                floatSetting("Afterglow Strength", key: "AS", range: 0...1, step: 0.01, get: { self.uniforms.AS }, set: { self.uniforms.AS = $0 }),
                floatSetting("Afterglow Saturation", key: "agsat", range: 0...2, step: 0.01, get: { self.uniforms.agsat }, set: { self.uniforms.agsat = $0 }),
                floatSetting("Display Gamut", key: "CS", range: 0...5, step: 1.0, get: { self.uniforms.CS }, set: { self.uniforms.CS = $0 }),
                floatSetting("CRT Profile", key: "CP", range: -1...5, step: 1.0, get: { self.uniforms.CP }, set: { self.uniforms.CP = $0 }),
                floatSetting("LUT Colors", key: "TNTC", range: 0...4, step: 1.0, get: { self.uniforms.TNTC }, set: { self.uniforms.TNTC = $0 }),
                floatSetting("LUT Size", key: "LS", range: 16...64, step: 16.0, get: { self.uniforms.LS }, set: { self.uniforms.LS = $0 }),
                floatSetting("Color Temperature", key: "WP", range: -100...100, step: 5.0, get: { self.uniforms.WP }, set: { self.uniforms.WP = $0 }),
                floatSetting("Saturation", key: "wp_saturation", range: 0...2, step: 0.05, get: { self.uniforms.wp_saturation }, set: { self.uniforms.wp_saturation = $0 }),
                floatSetting("Brightness", key: "pre_bb", range: 0...2, step: 0.01, get: { self.uniforms.pre_bb }, set: { self.uniforms.pre_bb = $0 }),
                floatSetting("Contrast", key: "contr", range: -2...2, step: 0.05, get: { self.uniforms.contr }, set: { self.uniforms.contr = $0 }),
                floatSetting("Gamma Correction", key: "pre_gc", range: 0.5...2, step: 0.01, get: { self.uniforms.pre_gc }, set: { self.uniforms.pre_gc = $0 }),
                floatSetting("Sega Brightness Fix", key: "sega_fix", range: 0...1, step: 1.0, get: { self.uniforms.sega_fix }, set: { self.uniforms.sega_fix = $0 }),
                floatSetting("Black Level", key: "BP", range: -100...25, step: 1.0, get: { self.uniforms.BP }, set: { self.uniforms.BP = $0 }),
                floatSetting("Vignette Strength", key: "vigstr", range: 0...2, step: 0.05, get: { self.uniforms.vigstr }, set: { self.uniforms.vigstr = $0 }),
                floatSetting("Vignette Size", key: "vigdef", range: 0.5...3, step: 0.1, get: { self.uniforms.vigdef }, set: { self.uniforms.vigdef = $0 }),
            ]),

            Group(title: "Raster Bloom", [
                floatSetting("Raster Bloom %", key: "BLOOM", range: -50...50, step: 1.0, get: { self.uniforms.BLOOM }, set: { self.uniforms.BLOOM = $0 }),
                floatSetting("Overscan Mode", key: "OS", range: 0...2, step: 1.0, get: { self.uniforms.OS }, set: { self.uniforms.OS = $0 }),
                floatSetting("Lum Smoothing", key: "lsmooth", range: 0.5...0.99, step: 0.01, get: { self.uniforms.lsmooth }, set: { self.uniforms.lsmooth = $0 }),
            ]),

            Group(title: "Gamma / Interlace", [
                floatSetting("Gamma Input", key: "GAMMA_INPUT", range: 1...5, step: 0.05, get: { self.uniforms.GAMMA_INPUT }, set: { self.uniforms.GAMMA_INPUT = $0 }),
                floatSetting("Gamma Output", key: "gamma_out", range: 1...5, step: 0.05, get: { self.uniforms.gamma_out }, set: { self.uniforms.gamma_out = $0 }),
                floatSetting("Interlace Trigger", key: "inter", range: 0...800, step: 25.0, get: { self.uniforms.inter }, set: { self.uniforms.inter = $0 }),
                floatSetting("Interlace Mode", key: "interm", range: 0...6, step: 1.0, get: { self.uniforms.interm }, set: { self.uniforms.interm = $0 }),
                floatSetting("Interlace Scanline", key: "iscan", range: 0...1, step: 0.05, get: { self.uniforms.iscan }, set: { self.uniforms.iscan = $0 }),
                floatSetting("Internal Resolution Y", key: "intres", range: 0...6, step: 0.5, get: { self.uniforms.intres }, set: { self.uniforms.intres = $0 }),
                floatSetting("Downsample Level X", key: "downsample_levelx", range: 0...3, step: 0.25, get: { self.uniforms.downsample_levelx }, set: { self.uniforms.downsample_levelx = $0 }),
                floatSetting("Downsample Level Y", key: "downsample_levely", range: 0...3, step: 0.25, get: { self.uniforms.downsample_levely }, set: { self.uniforms.downsample_levely = $0 }),
                floatSetting("Interlace Saturation", key: "iscans", range: 0...1, step: 0.05, get: { self.uniforms.iscans }, set: { self.uniforms.iscans = $0 }),
                floatSetting("VGA Mode", key: "vga_mode", range: 0...1, step: 1.0, get: { self.uniforms.vga_mode }, set: { self.uniforms.vga_mode = $0 }),
                floatSetting("Hi-Scan", key: "hiscan", range: 0...1, step: 1.0, get: { self.uniforms.hiscan }, set: { self.uniforms.hiscan = $0 }),
            ]),

            Group(title: "Magic Glow", [
                floatSetting("Glow Mode", key: "m_glow", range: 0...2, step: 1.0, get: { self.uniforms.m_glow }, set: { self.uniforms.m_glow = $0 }),
                floatSetting("Glow Cutoff", key: "m_glow_cutoff", range: 0...0.4, step: 0.01, get: { self.uniforms.m_glow_cutoff }, set: { self.uniforms.m_glow_cutoff = $0 }),
                floatSetting("Glow Low", key: "m_glow_low", range: 0...7, step: 0.05, get: { self.uniforms.m_glow_low }, set: { self.uniforms.m_glow_low = $0 }),
                floatSetting("Glow High", key: "m_glow_high", range: 0...7, step: 0.05, get: { self.uniforms.m_glow_high }, set: { self.uniforms.m_glow_high = $0 }),
                floatSetting("Glow Dist", key: "m_glow_dist", range: 0.2...4, step: 0.05, get: { self.uniforms.m_glow_dist }, set: { self.uniforms.m_glow_dist = $0 }),
                floatSetting("Glow Mask Str.", key: "m_glow_mask", range: 0...2, step: 0.025, get: { self.uniforms.m_glow_mask }, set: { self.uniforms.m_glow_mask = $0 }),
            ]),

            Group(title: "Glow", [
                floatSetting("Fine Glow", key: "FINE_GLOW", range: -3...3, step: 0.25, get: { self.uniforms.FINE_GLOW }, set: { self.uniforms.FINE_GLOW = $0 }),
                floatSetting("Glow H Size", key: "SIZEH", range: 0...50, step: 1.0, get: { self.uniforms.SIZEH }, set: { self.uniforms.SIZEH = $0 }),
                floatSetting("Glow H Sigma", key: "SIGMA_H", range: 0.2...15, step: 0.05, get: { self.uniforms.SIGMA_H }, set: { self.uniforms.SIGMA_H = $0 }),
                floatSetting("Glow V Size", key: "SIZEV", range: 0...50, step: 1.0, get: { self.uniforms.SIZEV }, set: { self.uniforms.SIZEV = $0 }),
                floatSetting("Glow V Sigma", key: "SIGMA_V", range: 0.2...15, step: 0.05, get: { self.uniforms.SIGMA_V }, set: { self.uniforms.SIGMA_V = $0 }),
                floatSetting("Glow Amount", key: "glow", range: -2...2, step: 0.01, get: { self.uniforms.glow }, set: { self.uniforms.glow = $0 }),
            ]),

            Group(title: "Bloom", [
                floatSetting("Fine Bloom", key: "FINE_BLOOM", range: -3...3, step: 0.25, get: { self.uniforms.FINE_BLOOM }, set: { self.uniforms.FINE_BLOOM = $0 }),
                floatSetting("Bloom H Size", key: "SIZEHB", range: 0...50, step: 1.0, get: { self.uniforms.SIZEHB }, set: { self.uniforms.SIZEHB = $0 }),
                floatSetting("Bloom H Sigma", key: "SIGMA_HB", range: 0.25...15, step: 0.05, get: { self.uniforms.SIGMA_HB }, set: { self.uniforms.SIGMA_HB = $0 }),
                floatSetting("Bloom V Size", key: "SIZEVB", range: 0...50, step: 1.0, get: { self.uniforms.SIZEVB }, set: { self.uniforms.SIZEVB = $0 }),
                floatSetting("Bloom V Sigma", key: "SIGMA_VB", range: 0.25...15, step: 0.05, get: { self.uniforms.SIGMA_VB }, set: { self.uniforms.SIGMA_VB = $0 }),
                floatSetting("Bloom Amount", key: "bloom", range: -2...2, step: 0.05, get: { self.uniforms.bloom }, set: { self.uniforms.bloom = $0 }),
                floatSetting("Bloom Dist", key: "bloom_dist", range: -2...3, step: 0.05, get: { self.uniforms.bloom_dist }, set: { self.uniforms.bloom_dist = $0 }),
                floatSetting("Halation", key: "halation", range: -2...2, step: 0.025, get: { self.uniforms.halation }, set: { self.uniforms.halation = $0 }),
                floatSetting("Mask Bloom", key: "mask_bloom", range: -2...2, step: 0.05, get: { self.uniforms.mask_bloom }, set: { self.uniforms.mask_bloom = $0 }),
                floatSetting("Bloom Mask", key: "bmask1", range: -1...1, step: 0.025, get: { self.uniforms.bmask1 }, set: { self.uniforms.bmask1 = $0 }),
                floatSetting("Halation Mask", key: "hmask1", range: 0...1, step: 0.025, get: { self.uniforms.hmask1 }, set: { self.uniforms.hmask1 = $0 }),
            ]),

            Group(title: "Brightness", [
                floatSetting("Gamma Scanline", key: "gamma_c", range: 0.5...2, step: 0.025, get: { self.uniforms.gamma_c }, set: { self.uniforms.gamma_c = $0 }),
                floatSetting("Gamma Mask", key: "gamma_c2", range: 0.5...2, step: 0.025, get: { self.uniforms.gamma_c2 }, set: { self.uniforms.gamma_c2 = $0 }),
                floatSetting("Bright Boost Dark", key: "brightboost", range: 0.25...10, step: 0.05, get: { self.uniforms.brightboost }, set: { self.uniforms.brightboost = $0 }),
                floatSetting("Bright Boost Bright", key: "brightboost1", range: 0.25...3, step: 0.025, get: { self.uniforms.brightboost1 }, set: { self.uniforms.brightboost1 = $0 }),
                floatSetting("Clips", key: "clips", range: -1...1, step: 0.05, get: { self.uniforms.clips }, set: { self.uniforms.clips = $0 }),
            ]),

            Group(title: "Scanline", [
                floatSetting("Scanline Type", key: "gsl", range: 0...2, step: 1.0, get: { self.uniforms.gsl }, set: { self.uniforms.gsl = $0 }),
                floatSetting("Scanline Beam Min", key: "scanline1", range: -20...40, step: 0.5, get: { self.uniforms.scanline1 }, set: { self.uniforms.scanline1 = $0 }),
                floatSetting("Scanline Beam Max", key: "scanline2", range: 0...70, step: 1.0, get: { self.uniforms.scanline2 }, set: { self.uniforms.scanline2 = $0 }),
                floatSetting("Beam Min", key: "beam_min", range: 0.25...10, step: 0.05, get: { self.uniforms.beam_min }, set: { self.uniforms.beam_min = $0 }),
                floatSetting("Beam Max", key: "beam_max", range: 0.2...3.5, step: 0.025, get: { self.uniforms.beam_max }, set: { self.uniforms.beam_max = $0 }),
                floatSetting("Thinner Dark Scanlines", key: "tds", range: 0...1, step: 1.0, get: { self.uniforms.tds }, set: { self.uniforms.tds = $0 }),
                floatSetting("Beam Size", key: "beam_size", range: 0...1, step: 0.05, get: { self.uniforms.beam_size }, set: { self.uniforms.beam_size = $0 }),
                floatSetting("Scanline Saturation", key: "scans", range: 0...2.5, step: 0.05, get: { self.uniforms.scans }, set: { self.uniforms.scans = $0 }),
                floatSetting("Scan Falloff", key: "scan_falloff", range: 0.1...2, step: 0.025, get: { self.uniforms.scan_falloff }, set: { self.uniforms.scan_falloff = $0 }),
                floatSetting("Spike", key: "spike", range: 0...2, step: 0.1, get: { self.uniforms.spike }, set: { self.uniforms.spike = $0 }),
                floatSetting("Scanline Gamma", key: "scangamma", range: 0.5...5, step: 0.05, get: { self.uniforms.scangamma }, set: { self.uniforms.scangamma = $0 }),
                floatSetting("Rolling Scan", key: "rolling_scan", range: -1...1, step: 0.01, get: { self.uniforms.rolling_scan }, set: { self.uniforms.rolling_scan = $0 }),
                floatSetting("No Scanlines", key: "no_scanlines", range: 0...1.5, step: 0.025, get: { self.uniforms.no_scanlines }, set: { self.uniforms.no_scanlines = $0 }),
            ]),

            Group(title: "Filtering", [
                floatSetting("Horizontal Sharpness", key: "h_sharp", range: 1...15, step: 0.05, get: { self.uniforms.h_sharp }, set: { self.uniforms.h_sharp = $0 }),
                floatSetting("Subpixel Sharpness", key: "s_sharp", range: 0...2, step: 0.05, get: { self.uniforms.s_sharp }, set: { self.uniforms.s_sharp = $0 }),
                floatSetting("Ringing", key: "ring", range: 0...3, step: 0.05, get: { self.uniforms.ring }, set: { self.uniforms.ring = $0 }),
                floatSetting("Smart Edge Inter.", key: "smart_ei", range: 0...0.5, step: 0.01, get: { self.uniforms.smart_ei }, set: { self.uniforms.smart_ei = $0 }),
                floatSetting("Smart EI Limit", key: "ei_limit", range: 0...0.75, step: 0.01, get: { self.uniforms.ei_limit }, set: { self.uniforms.ei_limit = $0 }),
                floatSetting("Smart EI Threshold", key: "sth", range: 0...1, step: 0.01, get: { self.uniforms.sth }, set: { self.uniforms.sth = $0 }),
            ]),

            Group(title: "Screen", [
                floatSetting("TATE Mode", key: "TATE", range: 0...1, step: 1.0, get: { self.uniforms.TATE }, set: { self.uniforms.TATE = $0 }),
                floatSetting("Integer Scaling", key: "IOS", range: 0...4, step: 1.0, get: { self.uniforms.IOS }, set: { self.uniforms.IOS = $0 }),
                floatSetting("CRT Curvature X", key: "warpX", range: 0...0.25, step: 0.01, get: { self.uniforms.warpX }, set: { self.uniforms.warpX = $0 }),
                floatSetting("CRT Curvature Y", key: "warpY", range: 0...0.25, step: 0.01, get: { self.uniforms.warpY }, set: { self.uniforms.warpY = $0 }),
                floatSetting("Curvature Shape", key: "c_shape", range: 0.05...0.6, step: 0.05, get: { self.uniforms.c_shape }, set: { self.uniforms.c_shape = $0 }),
                floatSetting("Overscan X", key: "overscanX", range: -200...200, step: 1.0, get: { self.uniforms.overscanX }, set: { self.uniforms.overscanX = $0 }),
                floatSetting("Overscan Y", key: "overscanY", range: -200...200, step: 1.0, get: { self.uniforms.overscanY }, set: { self.uniforms.overscanY = $0 }),
                floatSetting("Vertical Shift", key: "VShift", range: -100...100, step: 1.0, get: { self.uniforms.VShift }, set: { self.uniforms.VShift = $0 }),
            ]),

            Group(title: "Mask", [
                floatSetting("Mask Type", key: "shadowMask", range: -1...14, step: 1.0, get: { self.uniforms.shadowMask }, set: { self.uniforms.shadowMask = $0 }),
                floatSetting("Mask Strength", key: "maskstr", range: -0.5...1, step: 0.025, get: { self.uniforms.maskstr }, set: { self.uniforms.maskstr = $0 }),
                floatSetting("Mask Cutoff", key: "mcut", range: 0...2, step: 0.025, get: { self.uniforms.mcut }, set: { self.uniforms.mcut = $0 }),
                floatSetting("Mask Boost", key: "maskboost", range: 1...3, step: 0.05, get: { self.uniforms.maskboost }, set: { self.uniforms.maskboost = $0 }),
                floatSetting("Mask Size", key: "masksize", range: 1...4, step: 1.0, get: { self.uniforms.masksize }, set: { self.uniforms.masksize = $0 }),
                floatSetting("Mask Zoom", key: "mask_zoom", range: -5...5, step: 1.0, get: { self.uniforms.mask_zoom }, set: { self.uniforms.mask_zoom = $0 }),
                floatSetting("Mask Zoom Sharpness", key: "mzoom_sh", range: 0...2, step: 0.1, get: { self.uniforms.mzoom_sh }, set: { self.uniforms.mzoom_sh = $0 }),
                floatSetting("Mask Shift", key: "mshift", range: 0...8, step: 0.5, get: { self.uniforms.mshift }, set: { self.uniforms.mshift = $0 }),
                floatSetting("Mask Layout", key: "mask_layout", range: 0...1, step: 1.0, get: { self.uniforms.mask_layout }, set: { self.uniforms.mask_layout = $0 }),
                floatSetting("Mask Dark", key: "maskDark", range: 0...2, step: 0.05, get: { self.uniforms.maskDark }, set: { self.uniforms.maskDark = $0 }),
                floatSetting("Mask Light", key: "maskLight", range: 0...2, step: 0.05, get: { self.uniforms.maskLight }, set: { self.uniforms.maskLight = $0 }),
                floatSetting("Mask Gamma", key: "mask_gamma", range: 1...5, step: 0.025, get: { self.uniforms.mask_gamma }, set: { self.uniforms.mask_gamma = $0 }),
                floatSetting("Slot Mask", key: "slotmask", range: 0...1, step: 0.025, get: { self.uniforms.slotmask }, set: { self.uniforms.slotmask = $0 }),
                floatSetting("Slot Mask 2", key: "slotmask1", range: 0...1, step: 0.025, get: { self.uniforms.slotmask1 }, set: { self.uniforms.slotmask1 = $0 }),
                floatSetting("Slot Width", key: "slotwidth", range: 0...16, step: 1.0, get: { self.uniforms.slotwidth }, set: { self.uniforms.slotwidth = $0 }),
                floatSetting("Double Slot", key: "double_slot", range: 1...4, step: 1.0, get: { self.uniforms.double_slot }, set: { self.uniforms.double_slot = $0 }),
                floatSetting("Slot Mask Size", key: "slotms", range: 1...4, step: 1.0, get: { self.uniforms.slotms }, set: { self.uniforms.slotms = $0 }),
                floatSetting("Smooth Mask", key: "smoothmask", range: 0...1, step: 0.025, get: { self.uniforms.smoothmask }, set: { self.uniforms.smoothmask = $0 }),
                floatSetting("Smooth Mask Mitigate", key: "smask_mit", range: 0...1, step: 0.025, get: { self.uniforms.smask_mit }, set: { self.uniforms.smask_mit = $0 }),
                floatSetting("Base Mask", key: "bmask", range: 0...0.25, step: 0.01, get: { self.uniforms.bmask }, set: { self.uniforms.bmask = $0 }),
                floatSetting("Mask Clip", key: "mclip", range: 0...1, step: 0.025, get: { self.uniforms.mclip }, set: { self.uniforms.mclip = $0 }),
                floatSetting("Preserve Scanlines", key: "pr_scan", range: 0...1, step: 0.025, get: { self.uniforms.pr_scan }, set: { self.uniforms.pr_scan = $0 }),
                floatSetting("Edge Mask", key: "edgemask", range: 0...1, step: 0.05, get: { self.uniforms.edgemask }, set: { self.uniforms.edgemask = $0 }),
            ]),

            Group(title: "Border / Corner", [
                floatSetting("Corner Size", key: "csize", range: 0...0.25, step: 0.005, get: { self.uniforms.csize }, set: { self.uniforms.csize = $0 }),
                floatSetting("Border Size", key: "bsize1", range: 0...3, step: 0.01, get: { self.uniforms.bsize1 }, set: { self.uniforms.bsize1 = $0 }),
                floatSetting("Border Smoothness", key: "sborder", range: 0.25...2, step: 0.05, get: { self.uniforms.sborder }, set: { self.uniforms.sborder = $0 }),
            ]),

            Group(title: "Hum Bar", [
                floatSetting("Bar Speed", key: "barspeed", range: 5...200, step: 1.0, get: { self.uniforms.barspeed }, set: { self.uniforms.barspeed = $0 }),
                floatSetting("Bar Intensity", key: "barintensity", range: -1...1, step: 0.01, get: { self.uniforms.barintensity }, set: { self.uniforms.barintensity = $0 }),
                floatSetting("Bar Direction", key: "bardir", range: 0...1, step: 1.0, get: { self.uniforms.bardir }, set: { self.uniforms.bardir = $0 }),
            ]),

            Group(title: "Deconvergence", [
                floatSetting("DC Type X", key: "dctypex", range: 0...0.75, step: 0.05, get: { self.uniforms.dctypex }, set: { self.uniforms.dctypex = $0 }),
                floatSetting("DC Type Y", key: "dctypey", range: 0...0.75, step: 0.05, get: { self.uniforms.dctypey }, set: { self.uniforms.dctypey = $0 }),
                floatSetting("DC Shift Red X", key: "deconrr", range: -12...12, step: 0.25, get: { self.uniforms.deconrr }, set: { self.uniforms.deconrr = $0 }),
                floatSetting("DC Shift Green X", key: "deconrg", range: -12...12, step: 0.25, get: { self.uniforms.deconrg }, set: { self.uniforms.deconrg = $0 }),
                floatSetting("DC Shift Blue X", key: "deconrb", range: -12...12, step: 0.25, get: { self.uniforms.deconrb }, set: { self.uniforms.deconrb = $0 }),
                floatSetting("DC Shift Red Y", key: "deconrry", range: -12...12, step: 0.25, get: { self.uniforms.deconrry }, set: { self.uniforms.deconrry = $0 }),
                floatSetting("DC Shift Green Y", key: "deconrgy", range: -12...12, step: 0.25, get: { self.uniforms.deconrgy }, set: { self.uniforms.deconrgy = $0 }),
                floatSetting("DC Shift Blue Y", key: "deconrby", range: -12...12, step: 0.25, get: { self.uniforms.deconrby }, set: { self.uniforms.deconrby = $0 }),
                floatSetting("DC Strength", key: "decons", range: 0...1, step: 0.05, get: { self.uniforms.decons }, set: { self.uniforms.decons = $0 }),
            ]),

            Group(title: "Post / Noise", [
                floatSetting("Add Noise", key: "addnoised", range: -1...1, step: 0.02, get: { self.uniforms.addnoised }, set: { self.uniforms.addnoised = $0 }),
                floatSetting("Noise Resolution", key: "noiseresd", range: 1...10, step: 1.0, get: { self.uniforms.noiseresd }, set: { self.uniforms.noiseresd = $0 }),
                floatSetting("Noise Type", key: "noisetype", range: 0...1, step: 1.0, get: { self.uniforms.noisetype }, set: { self.uniforms.noisetype = $0 }),
                floatSetting("Post Brightness", key: "post_br", range: 0.25...5, step: 0.01, get: { self.uniforms.post_br }, set: { self.uniforms.post_br = $0 }),
                floatSetting("Original Image", key: "oimage", range: 0...1, step: 0.01, get: { self.uniforms.oimage }, set: { self.uniforms.oimage = $0 }),
            ]),
        ]
    }

    // Helper to create float settings concisely
    private func floatSetting(_ title: String, key: String,
                              range: ClosedRange<Double>, step: Float,
                              get: @escaping () -> Float,
                              set: @escaping (Float) -> Void) -> ShaderSetting {
        ShaderSetting(
            title: title,
            range: range, step: step,
            value: Binding(key: key, get: get, set: set))
    }

    override func revertToPreset(nr: Int) {
        uniforms = Uniforms.defaults
        frameCounter = 0
    }

    override func activate() {
        super.activate()

        afterglowKernel  = GuestKernel(name: "crtguest::guestAfterglowPass")
        preShadersKernel = GuestKernel(name: "crtguest::guestPreShadersPass")
        avgLumKernel     = GuestKernel(name: "crtguest::guestAvgLumPass")
        linearizeKernel  = GuestKernel(name: "crtguest::guestLinearizePass")
        gaussHKernel     = GuestKernel(name: "crtguest::guestGaussianHPass")
        gaussVKernel     = GuestKernel(name: "crtguest::guestGaussianVPass")
        bloomHKernel     = GuestKernel(name: "crtguest::guestBloomHPass")
        bloomVKernel     = GuestKernel(name: "crtguest::guestBloomVPass")
        crtKernel        = GuestKernel(name: "crtguest::guestCrtPass")
        deconvKernel     = GuestKernel(name: "crtguest::guestDeconvergencePass")

        // Load LUT textures
        loadLUTs()

        frameCounter = 0
    }

    override func retire() {
        super.retire()
        // Release intermediate textures
        afterglowTexA = nil; afterglowTexB = nil
        prePassTex = nil
        avgLumTexA = nil; avgLumTexB = nil
        linearizeTex = nil; gaussHTex = nil; glowTex = nil
        bloomHTex = nil; bloomTex = nil; crtTex = nil
        lut1 = nil; lut2 = nil; lut3 = nil; lut4 = nil
    }

    // MARK: - LUT loading

    private func loadLUTs() {

        let loader = MTKTextureLoader(device: ShaderLibrary.device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .origin: MTKTextureLoader.Origin.topLeft
        ]

        let lutFiles = [
            ("trinitron-lut", \CRTGuestAdvanced.lut1),
            ("inv-trinitron-lut", \CRTGuestAdvanced.lut2),
            ("nec-lut", \CRTGuestAdvanced.lut3),
            ("ntsc-lut", \CRTGuestAdvanced.lut4)
        ]

        for (name, keyPath) in lutFiles {
            if let url = Bundle.main.url(forResource: name, withExtension: "png") {
                do {
                    self[keyPath: keyPath] = try loader.newTexture(URL: url, options: options)
                    self[keyPath: keyPath]?.label = name
                } catch {
                    log("Failed to load LUT \(name): \(error)")
                }
            } else {
                log("LUT file not found: \(name).png")
            }
        }
    }

    // MARK: - Texture management

    func updateTextures(in input: MTLTexture, out output: MTLTexture) {
        let srcW = input.width
        let srcH = input.height
        let outW = output.width
        let outH = output.height

        let fmt16 = MTLPixelFormat.rgba16Float
        let fmtOut = output.pixelFormat

        // Source-size textures
        if afterglowTexA?.width != srcW || afterglowTexA?.height != srcH {
            afterglowTexA = Shader.makeTexture("afterglowA", width: srcW, height: srcH, pixelFormat: fmtOut)
            afterglowTexB = Shader.makeTexture("afterglowB", width: srcW, height: srcH, pixelFormat: fmtOut)
            prePassTex = Shader.makeTexture("prePass", width: srcW, height: srcH, pixelFormat: fmtOut)
            avgLumTexA = Shader.makeTexture("avgLumA", width: srcW, height: srcH, mipmaps: 8, pixelFormat: fmt16)
            avgLumTexB = Shader.makeTexture("avgLumB", width: srcW, height: srcH, mipmaps: 8, pixelFormat: fmt16)
            linearizeTex = Shader.makeTexture("linearize", width: srcW, height: srcH, pixelFormat: fmt16)
        }

        // Glow textures (at output size for proper full-screen blur)
        if gaussHTex?.width != outW || gaussHTex?.height != outH {
            gaussHTex = Shader.makeTexture("gaussH", width: outW, height: outH, pixelFormat: fmt16)
            glowTex = Shader.makeTexture("glow", width: outW, height: outH, pixelFormat: fmt16)
            bloomHTex = Shader.makeTexture("bloomH", width: outW, height: outH, pixelFormat: fmt16)
            bloomTex = Shader.makeTexture("bloom", width: outW, height: outH, pixelFormat: fmt16)
        }

        // Output-size textures
        if crtTex?.width != outW || crtTex?.height != outH {
            crtTex = Shader.makeTexture("crt", width: outW, height: outH, pixelFormat: fmt16)
        }
    }

    // MARK: - Apply

    override func apply(commandBuffer: MTLCommandBuffer,
                        in input: MTLTexture, out output: MTLTexture, rect: CGRect) {

        updateTextures(in: input, out: output)

        // Setup uniforms
        uniforms.sourceSize = SIMD2<Float>(Float(input.width), Float(input.height))
        uniforms.outputSize = SIMD2<Float>(Float(output.width), Float(output.height))
        uniforms.frameCount = frameCounter
        frameCounter &+= 1

        let stride = MemoryLayout<Uniforms>.stride

        // Select feedback buffers based on frame toggle
        let afterglowRead = feedbackFrame ? afterglowTexA! : afterglowTexB!
        let afterglowWrite = feedbackFrame ? afterglowTexB! : afterglowTexA!
        let avgLumRead = feedbackFrame ? avgLumTexA! : avgLumTexB!
        let avgLumWrite = feedbackFrame ? avgLumTexB! : avgLumTexA!
        feedbackFrame.toggle()

        // Pass 2: Afterglow
        afterglowKernel.apply(commandBuffer: commandBuffer,
                              textures: [input, afterglowRead, afterglowWrite],
                              options: &uniforms, length: stride)

        // Pass 3: Pre-shaders (color correction)
        // Note: LUT textures may be nil (no LUTs loaded), pass input as placeholder
        preShadersKernel.apply(commandBuffer: commandBuffer,
                               textures: [input, afterglowWrite,
                                          lut1 ?? input, lut2 ?? input,
                                          lut3 ?? input, lut4 ?? input,
                                          prePassTex!],
                               options: &uniforms, length: stride)

        // Pass 4: Average luminance (with feedback)
        // Generate mipmaps for the prepass so avgLum can sample at low LOD
        let mipEncoder = commandBuffer.makeBlitCommandEncoder()
        mipEncoder?.generateMipmaps(for: avgLumRead)
        mipEncoder?.endEncoding()

        avgLumKernel.apply(commandBuffer: commandBuffer,
                           textures: [prePassTex!, avgLumRead, avgLumWrite],
                           options: &uniforms, length: stride)

        // Pass 5: Linearize
        linearizeKernel.apply(commandBuffer: commandBuffer,
                              textures: [prePassTex!, linearizeTex!],
                              options: &uniforms, length: stride)

        // Pass 6: Gaussian horizontal (glow)
        gaussHKernel.apply(commandBuffer: commandBuffer,
                           textures: [linearizeTex!, gaussHTex!],
                           options: &uniforms, length: stride)

        // Pass 7: Gaussian vertical (glow)
        gaussVKernel.apply(commandBuffer: commandBuffer,
                           textures: [gaussHTex!, glowTex!],
                           options: &uniforms, length: stride)

        // Pass 8: Bloom horizontal
        bloomHKernel.apply(commandBuffer: commandBuffer,
                           textures: [linearizeTex!, bloomHTex!],
                           options: &uniforms, length: stride)

        // Pass 9: Bloom vertical
        bloomVKernel.apply(commandBuffer: commandBuffer,
                           textures: [bloomHTex!, bloomTex!],
                           options: &uniforms, length: stride)

        // Pass 10: CRT core
        crtKernel.apply(commandBuffer: commandBuffer,
                        textures: [linearizeTex!, avgLumWrite, prePassTex!, crtTex!],
                        options: &uniforms, length: stride)

        // Pass 11: Deconvergence / final composite
        deconvKernel.apply(commandBuffer: commandBuffer,
                           textures: [linearizeTex!, avgLumWrite, glowTex!, bloomTex!,
                                      prePassTex!, crtTex!, input, output],
                           options: &uniforms, length: stride)
    }
}

// MARK: - Kernel subclass

extension CRTGuestAdvanced {

    class GuestKernel: Kernel {
        convenience init?(name: String) {
            self.init(name: name, sampler: ShaderLibrary.linear)
        }
    }
}

// MARK: - Delegate

extension CRTGuestAdvanced: ShaderDelegate {

    func isHidden(setting: ShaderSetting) -> Bool { false }
    func settingDidChange(setting: ShaderSetting) { }
}
