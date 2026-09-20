// Crosshair + tooltip for every chart on the page. The chart is readable without
// this: each figure also ships a table view of the same numbers.
(function () {
  function fmt(value, spec) {
    var digits = parseInt((spec || '.4f').replace(/[^0-9]/g, ''), 10);
    return Number(value).toFixed(isNaN(digits) ? 4 : digits);
  }
  document.querySelectorAll('.chart').forEach(function (figure) {
    var raw = figure.getAttribute('data-hover');
    if (!raw) return;
    var data;
    try { data = JSON.parse(raw); } catch (e) { return; }
    var svg = figure.querySelector('svg');
    var hit = figure.querySelector('.hit');
    var cross = figure.querySelector('.crosshair');
    var tip = figure.querySelector('.tooltip');
    var plot = figure.querySelector('.plot');
    if (!svg || !hit || !cross || !tip || !plot || !data.px || !data.px.length) return;

    function nearest(viewX) {
      var best = 0, bestGap = Infinity;
      for (var i = 0; i < data.px.length; i++) {
        var gap = Math.abs(data.px[i] - viewX);
        if (gap < bestGap) { bestGap = gap; best = i; }
      }
      return best;
    }
    function show(event) {
      var box = svg.getBoundingClientRect();
      var scale = box.width / svg.viewBox.baseVal.width;
      var viewX = (event.clientX - box.left) / scale;
      var index = nearest(viewX);
      var x = data.px[index];
      cross.setAttribute('x1', x);
      cross.setAttribute('x2', x);
      cross.hidden = false;
      var rows = data.series.map(function (series, slot) {
        var value = series.values[index];
        if (value === undefined || value === null) return '';
        return '<div class="tt-row"><span><i class="series-' + (slot + 1) + '"></i>' +
          series.label + '</span><span class="tt-value">' + fmt(value, data.format) +
          '</span></div>';
      }).join('');
      tip.innerHTML = '<span class="tt-x">' + data.xLabel + ' ' + data.x[index] +
        '</span>' + rows;
      tip.hidden = false;
      var plotBox = plot.getBoundingClientRect();
      var left = x * scale;
      var width = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(left + 12, plotBox.width - width)) + 'px';
      tip.style.top = '8px';
    }
    function hide() { cross.hidden = true; tip.hidden = true; }
    hit.addEventListener('mousemove', show);
    hit.addEventListener('mouseleave', hide);
    hit.addEventListener('touchmove', function (event) {
      if (event.touches.length) show(event.touches[0]);
    }, { passive: true });
    hit.addEventListener('touchend', hide);
  });
})();
