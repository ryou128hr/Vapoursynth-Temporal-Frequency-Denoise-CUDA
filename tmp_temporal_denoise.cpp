#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>

// CUDA 関数を外部参照
extern "C" void runTemporalDenoise(
    const std::vector<const uint8_t*>& srcFrames,
    const std::vector<int>& strides,
    uint8_t* dst, int w, int h, int dstStride,
    int radius,
    float alphaLow, float alphaMid, float alphaHigh,
    float strength);

typedef struct {
    VSNode* node;
    VSVideoInfo vi;
    int radius;
    float alphaLow, alphaMid, alphaHigh;
    float strength;
} TDenoiseData;

// ------------------------------------
// フレーム取得
// ------------------------------------
static const VSFrame* VS_CC tdnGetFrame(
    int n, int activationReason, void* instanceData, void**,
    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    TDenoiseData* d = (TDenoiseData*)instanceData;

    if (activationReason == arInitial) {
        for (int t = -d->radius; t <= d->radius; t++) {
            int f = std::clamp(n + t, 0, d->vi.numFrames - 1);
            vsapi->requestFrameFilter(f, d->node, frameCtx);
        }
        return nullptr;
    }

    if (activationReason == arAllFramesReady) {
        std::vector<const VSFrame*> frames(2 * d->radius + 1);
        for (int t = -d->radius; t <= d->radius; t++) {
            int f = std::clamp(n + t, 0, d->vi.numFrames - 1);
            frames[t + d->radius] = vsapi->getFrameFilter(f, d->node, frameCtx);
        }

        const VSFrame* src = frames[d->radius];
        VSFrame* dst = vsapi->newVideoFrame(&d->vi.format,
            d->vi.width, d->vi.height,
            src, core);

        for (int plane = 0; plane < d->vi.format.numPlanes; plane++) {
            int w = vsapi->getFrameWidth(src, plane);
            int h = vsapi->getFrameHeight(src, plane);

            std::vector<const uint8_t*> refs;
            std::vector<int> strides;
            refs.reserve(frames.size());
            strides.reserve(frames.size());

            for (int i = 0; i < (int)frames.size(); i++) {
                refs.push_back(vsapi->getReadPtr(frames[i], plane));
                strides.push_back((int)vsapi->getStride(frames[i], plane));
            }

            uint8_t* dp = vsapi->getWritePtr(dst, plane);
            int dstStride = (int)vsapi->getStride(dst, plane);

            runTemporalDenoise(refs, strides, dp, w, h, dstStride,
                d->radius,
                d->alphaLow, d->alphaMid, d->alphaHigh,
                d->strength);
        }

        for (auto f : frames) vsapi->freeFrame(f);
        return dst;
    }

    return nullptr;
}

// ------------------------------------
// Free
// ------------------------------------
static void VS_CC tdnFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    TDenoiseData* d = (TDenoiseData*)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

// ------------------------------------
// Create
// ------------------------------------
static void VS_CC tdnCreate(const VSMap* in, VSMap* out, void* userData,
    VSCore* core, const VSAPI* vsapi)
{
    int err;
    TDenoiseData* d = (TDenoiseData*)malloc(sizeof(TDenoiseData));
    if (!d) {
        vsapi->mapSetError(out, "TemporalDenoise: malloc failed.");
        return;
    }

    d->node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err) {
        vsapi->mapSetError(out, "TemporalDenoise: clip required.");
        free(d);
        return;
    }

    d->vi = *vsapi->getVideoInfo(d->node);

    d->radius = (int)vsapi->mapGetInt(in, "radius", 0, &err); if (err) d->radius = 2;
    d->alphaLow = (float)vsapi->mapGetFloat(in, "alphaLow", 0, &err); if (err) d->alphaLow = 0.7f;
    d->alphaMid = (float)vsapi->mapGetFloat(in, "alphaMid", 0, &err); if (err) d->alphaMid = 0.4f;
    d->alphaHigh = (float)vsapi->mapGetFloat(in, "alphaHigh", 0, &err); if (err) d->alphaHigh = 0.1f;
    d->strength = (float)vsapi->mapGetFloat(in, "strength", 0, &err); if (err) d->strength = 1.0f;

    VSFilterDependency deps[] = { { d->node, rpGeneral } };
    vsapi->createVideoFilter(out, "TemporalDenoise", &d->vi,
        tdnGetFrame, tdnFree,
        fmParallel, deps, 1, d, core);
}

// ------------------------------------
// Init
// ------------------------------------
VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.example.temporaldenoise.cuda", "cuda_TMP",
        "Lightweight Temporal Denoiser (CUDA)",
        VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION,
        0, plugin);

    vspapi->registerFunction("TemporalDenoiseCUDA",
        "clip:vnode;radius:int:opt;alphaLow:float:opt;alphaMid:float:opt;alphaHigh:float:opt;strength:float:opt;",
        "clip:vnode;",
        tdnCreate, NULL, plugin);
}
