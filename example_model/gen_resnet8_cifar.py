"""
gen_resnet8_cifar.py
Export the OFFICIAL MLCommons/MLPerf-Tiny pretrained ResNet-8 (image
classification, CIFAR-10, FP32 Keras .h5) for the shared-FMA coprocessor
benchmark. No training - the network is used exactly as published.

What it does:
  1. loads pretrainedResnet.h5 (FP32 - asserted), maps layers to device roles,
  2. auto-detects the input preprocessing (the .h5 stores none) by evaluating
     candidates against CIFAR-10 test accuracy (expect ~85%),
  3. folds BatchNorm into the convs (standard inference transform; the network
     function is mathematically identical - handles conv bias present),
  4. flattens conv filters per output channel in (ky,kx,cin) order (device's
     HWC im2col patch order), FC out-major [M][N],
  5. SELF-CHECK: a numpy forward with EXACT device semantics (TF-'same'
     padding indices, patch order, folded weights) must match Keras logits ->
     any mapping/export bug fails HERE, not after 27h of RTL simulation,
  6. writes 8 normalized test images + labels + reference predictions and all
     weights (float bits as uint32) to a C header.

Run:
  ~/thesis/venv_thesis/bin/python example_model/gen_resnet8_cifar.py \
      --h5 example_model/pretrainedResnet.h5
Output: x-heep/sw/applications/perf_resnet8/resnet8_cifar_data.h
"""
import numpy as np, os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument('--h5', type=str,
                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'pretrainedResnet.h5'))
args = ap.parse_args()
np.random.seed(1)

# ---------------- official pretrained model (FP32) ----------------
model = tf.keras.models.load_model(args.h5, compile=False)
print(f"loaded {args.h5}: {model.count_params()} params")
for w in model.weights:
    assert w.dtype == tf.float32, f"non-FP32 weight found: {w.name} {w.dtype}"
print("all weights FP32 ✓")

# device-role -> layer-name map (verified against the eembc reference graph;
# note bn_4/bn_6 pair with the 3x3 convs, NOT the 1x1 skips created in between.
# The logit-level numpy self-check below fails loudly if this were ever wrong.)
names   = ['c1', 'b1c1', 'b1c2', 'b2c1', 'b2c2', 'b2s', 'b3c1', 'b3c2', 'b3s']
conv_of = {'c1':'conv2d','b1c1':'conv2d_1','b1c2':'conv2d_2',
           'b2c1':'conv2d_3','b2c2':'conv2d_4','b2s':'conv2d_5',
           'b3c1':'conv2d_6','b3c2':'conv2d_7','b3s':'conv2d_8'}
bn_of   = {'c1':'batch_normalization','b1c1':'batch_normalization_1',
           'b1c2':'batch_normalization_2','b2c1':'batch_normalization_3',
           'b2c2':'batch_normalization_4','b2s':None,
           'b3c1':'batch_normalization_5','b3c2':'batch_normalization_6','b3s':None}
FC_NAME = 'dense'

# ---------------- CIFAR-10 + preprocessing auto-detection ----------------
(xtr_r, ytr), (xte_r, yte) = tf.keras.datasets.cifar10.load_data()
ytr = ytr.flatten(); yte = yte.flatten()
xtr_r = xtr_r.astype(np.float32); xte_r = xte_r.astype(np.float32)
mean = (xtr_r/255.).mean(axis=(0, 1, 2)); std = (xtr_r/255.).std(axis=(0, 1, 2))
pixmean = (xtr_r/255.).mean(axis=0)
cands = {
    'div255'        : lambda x: x/255.,
    'div255_meanstd': lambda x: (x/255. - mean)/std,
    'div255_pixmean': lambda x: x/255. - pixmean,
    'raw0_255'      : lambda x: x,
}
sub, ysub = xte_r[:2000], yte[:2000]
accs = {}
for nm, f in cands.items():
    p = model.predict(f(sub).astype(np.float32), batch_size=256, verbose=0).argmax(1)
    accs[nm] = float((p == ysub).mean())
    print(f"  preproc {nm:16s} -> acc {accs[nm]*100:.2f}%")
best = max(accs, key=accs.get)
assert accs[best] > 0.70, f"no preprocessing candidate reaches sane accuracy (best {best}={accs[best]:.2f})"
PRE = cands[best]
print(f"chosen preprocessing: {best}")
xte = PRE(xte_r).astype(np.float32)
acc = float((model.predict(xte, batch_size=256, verbose=0).argmax(1) == yte).mean())
print(f"\n=== Full CIFAR-10 test accuracy (official pretrained, FP32): {acc*100:.2f}% ===")

# ---------------- fold BN into convs (conv bias handled) ----------------
# BN(Wx+b0) = (sc*W)x + [beta + (b0-mu)*sc],   sc = gamma/sqrt(var+eps)
folded = {}
for nm in names:
    cv = model.get_layer(conv_of[nm]); wts = cv.get_weights()
    K = wts[0]; b0 = wts[1] if len(wts) > 1 else np.zeros(K.shape[-1], np.float32)
    if bn_of[nm]:
        bn = model.get_layer(bn_of[nm])
        g, be, mu, var = bn.get_weights()
        sc = g / np.sqrt(var + bn.epsilon)
        assert sc.shape[0] == K.shape[-1], f"BN/conv channel mismatch at {nm}"
        Wf = K * sc.reshape(1, 1, 1, -1); bf = be + (b0 - mu) * sc
    else:
        Wf, bf = K, b0
    kh, kw, ci, co = Wf.shape
    # flatten per output channel in (ky,kx,ci) order -> [co][kh*kw*ci]
    folded[nm] = (np.transpose(Wf, (3, 0, 1, 2)).reshape(co, kh*kw*ci).astype(np.float32),
                  bf.astype(np.float32))
Wfc, bfc = model.get_layer(FC_NAME).get_weights()      # (64,10) -> out-major [10][64]
folded['fc'] = (Wfc.T.astype(np.float32).copy(), bfc.astype(np.float32))

# ---------------- self-check: numpy forward with EXACT device semantics ----------------
def tf_same(n, k, s):                                  # TF 'same': (out, pad_begin)
    o = -(-n // s); p = max((o-1)*s + k - n, 0); return o, p // 2
def conv_np(a, nm, k, s, relu):                        # a: [H][W][C] (HWC)
    W, b = folded[nm]; H, Wd, C = a.shape; co = W.shape[0]
    O, pb = tf_same(H, k, s)
    out = np.zeros((O, O, co), np.float32)
    for yy in range(O):
        for xx in range(O):
            patch = np.zeros(k*k*C, np.float32); i = 0
            for ky in range(k):
                iy = yy*s + ky - pb
                for kx in range(k):
                    ix = xx*s + kx - pb
                    if 0 <= iy < H and 0 <= ix < Wd:
                        patch[i:i+C] = a[iy, ix, :]
                    i += C
            out[yy, xx, :] = W @ patch + b
    return np.maximum(out, 0) if relu else out
def fwd_np(img):
    x = conv_np(img, 'c1', 3, 1, True)
    y = conv_np(conv_np(x, 'b1c1', 3, 1, True), 'b1c2', 3, 1, False)
    x = np.maximum(x + y, 0)
    y = conv_np(conv_np(x, 'b2c1', 3, 2, True), 'b2c2', 3, 1, False)
    x = np.maximum(conv_np(x, 'b2s', 1, 2, False) + y, 0)
    y = conv_np(conv_np(x, 'b3c1', 3, 2, True), 'b3c2', 3, 1, False)
    x = np.maximum(conv_np(x, 'b3s', 1, 2, False) + y, 0)
    return folded['fc'][0] @ x.mean(axis=(0, 1)) + folded['fc'][1]

NTEST = 8
imgs, labs = xte[:NTEST], yte[:NTEST]
keras_out = model.predict(imgs, verbose=0)      # NOTE: official Dense has softmax -> probabilities
np_log = np.stack([fwd_np(imgs[j]) for j in range(NTEST)])
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)
np_prob = softmax(np_log)                       # device emits logits; argmax identical either way
keras_pred, np_pred = keras_out.argmax(1), np_log.argmax(1)
print("keras preds :", keras_pred.tolist())
print("numpy(dev)  :", np_pred.tolist(), " labels:", labs.tolist())
assert np.allclose(np_prob, keras_out, rtol=5e-2, atol=1e-3), \
    f"fold/mapping self-check FAILED (max |prob diff| = {np.abs(np_prob-keras_out).max()})"
assert (keras_pred == np_pred).all()
print("self-check OK: folded numpy forward (device semantics) == keras (softmax-prob level)")

# ---------------- emit C header ----------------
outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'x-heep', 'sw', 'applications', 'perf_resnet8', 'resnet8_cifar_data.h')
os.makedirs(os.path.dirname(outp), exist_ok=True)
def bits(a): return np.ascontiguousarray(a.astype(np.float32)).flatten().view(np.uint32)
with open(outp, 'w') as f:
    f.write("// AUTO-GENERATED by gen_resnet8_cifar.py\n")
    f.write("// OFFICIAL MLCommons/MLPerf-Tiny pretrained ResNet-8 (CIFAR-10), FP32, BN folded.\n")
    f.write(f"// Filters [co][(ky,kx,ci)]. TF-'same' padding. Preproc: {best}. Test acc: {acc*100:.2f}%\n")
    f.write("#ifndef RESNET8_CIFAR_DATA_H\n#define RESNET8_CIFAR_DATA_H\n\n")
    f.write(f"#define R8_NTEST {NTEST}\n\n")
    def arr(name, u):
        f.write(f"static const unsigned int {name}[{len(u)}] = {{\n")
        for i in range(0, len(u), 12):
            f.write("  " + ",".join(f"0x{v:08x}" for v in u[i:i+12]) + ",\n")
        f.write("};\n\n")
    for nm in names + ['fc']:
        W, b = folded[nm]
        arr(f"{nm}_w", bits(W)); arr(f"{nm}_b", bits(b))
    arr("test_images", bits(imgs))                     # [NTEST][32][32][3] HWC, normalized
    f.write(f"static const int test_labels[{NTEST}] = {{{','.join(map(str, labs.tolist()))}}};\n")
    f.write(f"static const int ref_pred[{NTEST}]    = {{{','.join(map(str, keras_pred.tolist()))}}};\n")
    f.write("\n#endif\n")
print(f"\nwrote {outp}")
print(f"params: {sum(w.size + b.size for w, b in folded.values())} "
      f"({sum(w.size + b.size for w, b in folded.values())*4/1024:.0f} KB FP32)")
