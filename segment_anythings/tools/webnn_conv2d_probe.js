// Paste into DevTools Console BEFORE the page builds its WebNN graph.
//
// Logs every MLGraphBuilder.conv2d call and, on the one that throws, prints the
// exact numbers WebNN validated so you can see which of the two conditions in
// services/webnn/public/cpp/graph_validation_utils.cc:781-787 failed:
//
//     input_channels % groups != 0  ||
//     filter_input_channels != input_channels / groups
//
// Filter-layout -> filter_input_channels axis (graph_validation_utils.cc:745-773):
//     hwio [h, w, ic/g, oc]  -> shape[2]
//     ohwi [oc, h, w, ic/g]  -> shape[3]
//     ihwo [ic/g, h, w, oc]  -> shape[0]
//     oihw [oc, ic/g, h, w]  -> shape[1]   (WebNN default)

(() => {
  const FILTER_IC_AXIS = { hwio: 2, ohwi: 3, ihwo: 0, oihw: 1 };
  const orig = MLGraphBuilder.prototype.conv2d;
  let n = 0;
  window.__conv2d = [];

  MLGraphBuilder.prototype.conv2d = function (input, filter, options) {
    const o = options || {};
    const rec = {
      i: n++,
      input: [...input.shape],
      filter: [...filter.shape],
      groups: o.groups ?? 1,
      inputLayout: o.inputLayout ?? 'nchw',   // WebNN IDL defaults
      filterLayout: o.filterLayout ?? 'oihw',
      label: o.label ?? '',
    };
    window.__conv2d.push(rec);
    try {
      return orig.call(this, input, filter, options);
    } catch (e) {
      const ic = rec.inputLayout === 'nhwc' ? rec.input[3] : rec.input[1];
      const axis = FILTER_IC_AXIS[rec.filterLayout];
      const fic = rec.filter[axis];
      console.error(`conv2d #${rec.i} THREW: ${e.message}`);
      console.error('  call    ', rec);
      console.error(`  input_channels        = ${ic}`);
      console.error(`  filter_input_channels = ${fic}  (${rec.filterLayout}[${axis}])`);
      console.error(`  groups passed         = ${rec.groups}`);
      console.error(`  ic % groups           = ${ic % rec.groups}   (must be 0)`);
      console.error(`  ic / groups           = ${ic / rec.groups}   (must equal ${fic})`);
      console.error(`  => groups SHOULD be ${ic / fic}`);
      throw e;
    }
  };
  console.log('conv2d hooked; window.__conv2d collects every call');
})();
