"""Exact structural spatial dependencies of the fixed depth-2 UNet2 ablation.

Tracks sets of reachable *real input* indices through every spatial operation.
Convolution padding is constant zero and is therefore not an input dependency.
MaxPool tracks every potential input, not the data-dependent winning index.
Bilinear align_corners=False is propagated with exact rational coordinates.
This is a structural/theoretical receptive field, not a gradient ERF estimate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def conv_support(axis, dilation=1):
    """Stride-one, same-shape kernel-3 convolution with padding=dilation."""
    n = len(axis)
    return [set().union(*(axis[j] for j in (i-dilation, i, i+dilation)
                         if 0 <= j < n)) for i in range(n)]


def pool_support(axis):
    """MaxPool2d(kernel_size=2,stride=2,ceil_mode=False), along one axis."""
    return [axis[2*i] | axis[2*i+1] for i in range(len(axis)//2)]


def interpolate_support(axis, output_size):
    """F.interpolate(mode='bilinear',size=...,align_corners=False), one axis.

    Source coordinate = (i+0.5)*input_size/output_size - 0.5.
    A zero interpolation weight creates no dependency. Edge coordinates clamp.
    Rational arithmetic avoids treating a numerically tiny weight as nonzero.
    """
    n = len(axis)
    out = []
    denominator = 2*output_size
    for i in range(output_size):
        numerator = (2*i+1)*n-output_size
        lower, remainder = divmod(numerator, denominator)
        indices = [max(0, min(n-1, lower))]
        if remainder:
            indices.append(max(0, min(n-1, lower+1)))
        out.append(set().union(*(axis[j] for j in indices)))
    return out


def support_axis(input_size, bottleneck_dilations=(1, 1)):
    """Match all operations and ordering of UNet2(...,k3,depth2,b128)."""
    if input_size < 4:
        raise ValueError('Depth-2 U-Net requires each axis to have size >=4')
    h = [{i} for i in range(input_size)]
    enc0 = conv_support(conv_support(h))
    h = conv_support(pool_support(enc0))
    enc1 = conv_support(conv_support(h))
    h = conv_support(pool_support(enc1))
    h = conv_support(h, bottleneck_dilations[0])
    h = conv_support(h, bottleneck_dilations[1])
    for skip in (enc1, enc0):
        h = conv_support(h)  # self.ups conv PRECEDES interpolation
        h = interpolate_support(h, len(skip))
        h = [deep | short for deep, short in zip(h, skip)]
        h = conv_support(conv_support(h))
    return conv_support(h)  # final kernel-3 head


def summarize_axis(supports):
    middle = len(supports)//2
    def row(i, values):
        lo, hi = min(values), max(values)
        return {'output_index': i, 'input_min': lo, 'input_max': hi,
                'support_count': len(values), 'bounding_span': hi-lo+1,
                'contiguous': len(values) == hi-lo+1}
    rows = [row(i, values) for i, values in enumerate(supports)]
    return {'input_size': len(supports), 'center': rows[middle],
            'max_support_count': max(r['support_count'] for r in rows),
            'max_bounding_span': max(r['bounding_span'] for r in rows),
            'all_supports_contiguous': all(r['contiguous'] for r in rows)}, rows


def theoretical_report(height=97, width=241, probe_size=513):
    # Odd probe size preserves floor-pool/odd-resize behavior and leaves an
    # interior region wider than both models' structural receptive fields.
    if probe_size < 257 or probe_size % 4 != 1:
        raise ValueError('Use an unclipped probe >=257 with size %4 ==1')
    report = {
        'method': 'exact discrete structural dependency propagation',
        'architecture': 'depth2 UNet2, all kernels3, pool2, pre-interpolation up conv, bilinear align_corners=False, skips, kernel3 head',
        'input_shape_spatial': [height, width],
        'scope': 'theoretical potential input dependencies; ignores ReLU gates, maxpool winner selection, trained weights and gradient magnitude',
        'units': 'input-grid pixels, not kilometres or independent observations',
        'padding': 'constant-zero positions excluded from real-input support',
        'models': {},
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    all_rows = []
    for name, dilations in [('original', (1, 1)), ('dilated', (2, 3))]:
        model = {'bottleneck_dilations': list(dilations)}
        for axis_name, size in [('height', height), ('width', width),
                                ('unclipped_probe', probe_size)]:
            supports = support_axis(size, dilations)
            summary, rows = summarize_axis(supports)
            model[axis_name] = summary
            all_rows.extend({'model': name, 'axis': axis_name, **r} for r in rows)
        model['center_bounding_box_hw'] = [model[a]['center']['bounding_span']
                                           for a in ('height', 'width')]
        model['max_actual_axis_spans_hw'] = [model[a]['max_bounding_span']
                                            for a in ('height', 'width')]
        model['unclipped_max_axis_span'] = model['unclipped_probe']['max_bounding_span']
        report['models'][name] = model
    return report, all_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, help='Fresh output directory; omit for JSON stdout only')
    a = p.parse_args()
    report, rows = theoretical_report()
    assert report['models']['original']['center_bounding_box_hw'] == [72, 72]
    assert report['models']['dilated']['center_bounding_box_hw'] == [96, 96]
    assert report['models']['original']['unclipped_max_axis_span'] == 76
    assert report['models']['dilated']['unclipped_max_axis_span'] == 100
    if a.out:
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out/'theoretical_rf.json').write_text(json.dumps(report, indent=2))
        with (a.out/'theoretical_rf_all_coordinates.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
