/* Restore this application's own theme before first paint. */
(function () {
  var root = document.documentElement;
  var preference = matchMedia('(prefers-color-scheme: dark)');
  try {
    var saved = localStorage.getItem('rangamati-theme');
    if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
  } catch (_) {}
  function ready() {
    var button = document.getElementById('themer');
    function dark() { return root.dataset.theme ? root.dataset.theme === 'dark' : preference.matches; }
    function update() {
      if (!button) return;
      button.textContent = dark() ? 'Light' : 'Dark';
      button.setAttribute('aria-label', 'Switch to ' + (dark() ? 'light' : 'dark') + ' theme');
    }
    if (button) button.addEventListener('click', function () {
      root.dataset.theme = dark() ? 'light' : 'dark';
      try { localStorage.setItem('rangamati-theme', root.dataset.theme); } catch (_) {}
      update();
    });
    preference.addEventListener('change', update);
    update();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ready);
  else ready();
})();
