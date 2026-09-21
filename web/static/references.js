(function () {
  var target = null;
  var configured = window.RANGAMATI_CONFIG && window.RANGAMATI_CONFIG.tosReferenceUrl;
  try {
    if (configured) {
      var url = new URL(configured);
      if (['https:', 'http:'].includes(url.protocol)) target = url.href;
    }
  } catch (_) {}
  if (!target && ['localhost', '127.0.0.1'].includes(location.hostname)) {
    target = 'http://' + location.hostname + ':4173/research.html#tos';
  }
  if (!target) return;
  document.querySelectorAll('[data-tos-reference]').forEach(function (link) {
    link.href = target;
    link.hidden = false;
  });
})();
