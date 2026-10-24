#!/usr/bin/env python3
"""Exact GGUF tensor inventory (spec-correct, no external deps).

Usage: python3 gguf_inventory.py <file.gguf> [--markdown out.md] [--csv out.csv]

For every tensor: name, ggml type, dims (ggml order: ne0 = contiguous/row length),
absolute file offset, exact byte size computed from the ggml block layout, and a
check that size <= distance to the next tensor (alignment padding only).
"""
import struct
import sys
from collections import defaultdict

# ggml_type -> (name, block_elems, block_bytes)   (ggml/include/ggml.h, ggml-common.h)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


def parse(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, = struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        assert n < (1 << 30)
        return f.read(n).decode("utf-8", "replace")

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
              10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        if t in scalar:
            fmt = scalar[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            cnt, = struct.unpack("<Q", f.read(8))
            if et in scalar:  # bulk read
                fmt = scalar[et]
                sz = struct.calcsize(fmt)
                raw = f.read(sz * cnt)
                return list(struct.unpack("<%d%s" % (cnt, fmt[1]), raw))
            return [rval(et) for _ in range(cnt)]
        raise ValueError("bad type %d" % t)

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        t, = struct.unpack("<I", f.read(4))
        kv[k] = rval(t)
    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
        typ, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        tensors.append(dict(name=name, dims=dims, type=typ, rel=off))
    align = kv.get("general.alignment", 32)
    meta_end = f.tell()
    data_start = (meta_end + align - 1) // align * align
    fsize = f.seek(0, 2)
    for t in tensors:
        tn, be, bb = GGML_TYPES[t["type"]]
        ne = 1
        for d in t["dims"]:
            ne *= d
        assert t["dims"][0] % be == 0, t
        t["tname"] = tn
        t["nelem"] = ne
        t["nbytes"] = ne // be * bb
        t["abs"] = data_start + t["rel"]
        assert t["rel"] % align == 0, ("misaligned", t["name"])
    order = sorted(tensors, key=lambda t: t["rel"])
    for i, t in enumerate(order):
        nxt = order[i + 1]["abs"] if i + 1 < len(order) else fsize
        t["gap"] = nxt - t["abs"] - t["nbytes"]
        assert 0 <= t["gap"] < align, ("overlap/gap", t["name"], t["gap"])
    return dict(version=version, kv=kv, tensors=tensors, data_start=data_start,
                meta_end=meta_end, fsize=fsize, align=align)


def main():
    path = sys.argv[1]
    md = sys.argv[sys.argv.index("--markdown") + 1] if "--markdown" in sys.argv else None
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    g = parse(path)
    T = g["tensors"]
    print("GGUF v%d, %d KV, %d tensors, meta_end=0x%x data_start=0x%x file=%d" %
          (g["version"], len(g["kv"]), len(T), g["meta_end"], g["data_start"], g["fsize"]))
    by_type = defaultdict(lambda: [0, 0, 0])
    for t in T:
        e = by_type[t["tname"]]
        e[0] += 1
        e[1] += t["nelem"]
        e[2] += t["nbytes"]
    tot = sum(t["nbytes"] for t in T)
    print("total tensor bytes: %d (%.3f GiB), data section %d" % (tot, tot / 2**30, g["fsize"] - g["data_start"]))
    for k, (n, ne, nb) in sorted(by_type.items(), key=lambda x: -x[1][2]):
        print("  %-8s %4d tensors %14d elems %13d bytes (%6.1f MiB) %.4f bpw" % (k, n, ne, nb, nb / 2**20, nb * 8 / ne))
    # per-role summary
    role = defaultdict(lambda: defaultdict(int))
    for t in T:
        n = t["name"]
        r = n.split(".", 2)[2] if n.startswith("blk.") else n
        role[r][t["tname"]] += 1
    print("\nper-role type histogram:")
    for r in sorted(role):
        tb = sum(t["nbytes"] for t in T if (t["name"].split(".", 2)[2] if t["name"].startswith("blk.") else t["name"]) == r)
        print("  %-34s %8.1f MiB  %s" % (r, tb / 2**20, dict(role[r])))
    lines = []
    for t in sorted(T, key=lambda t: t["rel"]):
        lines.append((t["name"], t["tname"], "x".join(map(str, t["dims"])), t["abs"], t["nbytes"], t["gap"]))
    if csv:
        with open(csv, "w") as fo:
            fo.write("name,type,dims,abs_offset,nbytes,pad\n")
            for l in lines:
                fo.write("%s,%s,%s,%d,%d,%d\n" % l)
    if md:
        with open(md, "w") as fo:
            fo.write("| # | tensor | type | dims (ne0 x ne1) | abs offset | bytes |\n|---|---|---|---|---|---|\n")
            for i, l in enumerate(lines):
                fo.write("| %d | `%s` | %s | %s | 0x%x | %d |\n" % (i, l[0], l[1], l[2], l[3], l[4]))


if __name__ == "__main__":
    main()
