/* Rangamati, as a thing you can walk on.
 *
 * A susceptibility map is an image. You look at it, you agree that red is worse than green, and
 * nothing about the number under any particular pixel ever becomes concrete. This is the same
 * grid as a solid: 5,067 cells of the district at 0.01 degrees, lifted by the elevation SRTM
 * reports for each one, coloured by the probability the selected model assigns it, and given a
 * surveyor who can be sent across it.
 *
 * The rule is the model's own classification, not something invented for the game. Ground the
 * model calls High (0.60 to 0.80) shakes underfoot. Ground it calls Very high (0.80 and up)
 * gives way, and the surveyor goes down the slope with it. Every fall leaves a scar in the mesh
 * at the cell that failed, so the map slowly accumulates a record of which cells the model was
 * most sure about. That is the honest reading of a susceptibility map: it is a ranking of where
 * you would rather not be standing.
 *
 * Vertical scale is exaggerated 7x. The district is 210 km from end to end and 895 m at its
 * highest point, so at true scale it is a sheet of paper and there is nothing to walk over.
 * 7x was chosen by measuring the grid rather than by eye: it puts the 99th percentile step
 * between neighbouring cells at about 62 degrees, which reads as hill country. At 22x, the
 * first thing I tried, that same step is 80 degrees and the district is a bed of nails.
 * Everything else here is at its real size and its real value.
 *
 * Usage. Either mark the host element up and let it start itself:
 *
 *     <div id="walk" data-walk="/static/walk_data.js" data-walk-hud="walkHud"></div>
 *
 * or drive it, which is what the prediction app does, so that a click on its map and a row of
 * its inventory both move the surveyor:
 *
 *     var w = RangamatiWalk.init({ host: "walk", data: "/static/walk_data.js" });
 *     w.moveTo(22.65, 92.17, "Measured here.");     // a real coordinate
 *     w.placeByClass(26, 22, "Ground like this.");  // an inventory row, which has no coordinate
 *
 * Renders only where WebGL is available; the host degrades to a line of text otherwise.
 */
window.RangamatiWalk = (function () {
  "use strict";

  function init(opts) {
    opts = opts || {};

    function el(v) {
      return typeof v === "string" ? document.getElementById(v) : (v || null);
    }

    var host = el(opts.host);
    if (!host) return null;

    var hud = el(opts.hud);
    var banner = el(opts.banner);
    var intro = el(opts.intro);
    var startBtn = el(opts.start);
    var dataUrl = opts.data || "walk_data.js";
    var onArrive = opts.onArrive || null;

    /* The host page owns the look of these two; this module only ever adds a state class on
       top of whatever it finds. Rebuilding className from a literal here is what broke the
       readout in the prediction app: it quietly renamed `walk-hud` to `game-hud`, and the
       element lost every rule the page had given it. */
    var hudBase = hud ? hud.className : "";
    var bannerBase = banner ? banner.className : "";

    /* ------------------------------------------------------------------ constants */

    var VEX = 7;               // vertical exaggeration; see the note below
    var KM_LAT = 1.11;         // one 0.01 degree step in latitude, kilometres
    var KM_LON = 1.03;         // and in longitude at 22.5 N
    var SPEED = 9;             // km per second, which is not realistic and is the point
      /* Long enough to dash across a narrow red neck, short enough that a broad one is fatal.
       At 9 km/s this buys about four cells of Very high ground, and roughly a quarter of the
       district scores that badly, so the safe routes are the valleys the model already likes. */
    var FUSE = 0.45;
    var DEPTH = 6;             // how thick the slab looks from the side

    /* The five classes of the printed map, and the colours it uses for them. */
    var BANDS = [[0.20, 0x4E7A55], [0.40, 0x7C9A55], [0.60, 0xC79B33],
                 [0.80, 0xC26A32], [1.01, 0xA8452C]];
    var NAMES = ["Very low", "Low", "Moderate", "High", "Very high"];
    var BAND_HEX = ["#4E7A55", "#7C9A55", "#C79B33", "#C26A32", "#A8452C"];

    function bandIndex(p) {
      for (var i = 0; i < BANDS.length; i++) if (p < BANDS[i][0]) return i;
      return 4;
    }

    /* ------------------------------------------------------------------ data */

    function unpack(b64) {
      var bin = atob(b64), n = bin.length, out = new Uint8Array(n);
      for (var i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
      return out;
    }

    var D, NROW, NCOL, elevB, probB, slopeB, maskB, eMin, eSpan, sMax;

    function elevAt(i, j) {                      // metres, by cell index
      return eMin + (elevB[j * NCOL + i] / 255) * eSpan;
    }
    function yAt(i, j) {                         // world units
      return (elevAt(i, j) / 1000) * VEX;
    }
    function wx(i) { return (i - (NCOL - 1) / 2) * KM_LON; }
    function wz(j) { return -(j - (NROW - 1) / 2) * KM_LAT; }

    /* ------------------------------------------------------------------ bootstrap */

    var booted = false, ready = false, pending = null;

    function needData(then) {
      if (window.RANGAMATI) { then(); return; }
      var s = document.createElement("script");
      s.src = dataUrl;
      s.onload = then;
      s.onerror = function () { fail("The terrain data did not load."); };
      document.head.appendChild(s);
    }

    function fail(msg) {
      host.innerHTML = '<p class="walk-fail">' + msg +
        " The figure above the caption is the same map, drawn flat.</p>";
      if (intro) intro.hidden = true;
    }

    function boot() {
      if (booted) return;
      booted = true;
      if (!window.THREE) { fail("This needs WebGL, which this browser is not offering."); return; }
      needData(build);
    }

    /* Only pay for 70 KB of terrain and a WebGL context if the section is actually reached. */
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (es, o) {
        if (es[0].isIntersecting) { o.disconnect(); boot(); }
      }, { rootMargin: "300px" }).observe(host);
    } else {
      boot();
    }

    /* ------------------------------------------------------------------ the scene */

    var renderer, scene, camera, terrain, topGeom, topCol, surveyor, legL, legR;
    var summit = null, summitFlag = null, summitCloth = null, summitMet = false;
    var fellAt = null;
    var raf = null, reduced = false, started = false, active = false;
    var state = "walk", fuse = 0, fallT = 0, shake = 0, grace = false, placed = false;
    var walked = 0, falls = 0, safeSpot = null, gradeDir;
    var keys = {}, phase = 0, facing = 0, clock = 0;
    var gestureId = null, gestureX = 0, gestureY = 0, gestureOriginX = 0, gestureOriginY = 0;
    var gestureRing, gestureThumb;
    var camPos, camAim, camTarget, camLook;

    function build() {
      /* Allocated here rather than at the top of the file: this module is parsed even when
         three.js failed to load, and touching THREE before the guard would take the page down. */
      gradeDir = new THREE.Vector3();
      camPos = new THREE.Vector3(-46, 262, 55);
      camAim = new THREE.Vector3();
      camTarget = new THREE.Vector3();
      camLook = new THREE.Vector3();

      D = window.RANGAMATI;
      NROW = D.nrow; NCOL = D.ncol;
      elevB = unpack(D.elev); probB = unpack(D.prob);
      slopeB = unpack(D.slope); maskB = unpack(D.mask);
      eMin = D.elev_min_m; eSpan = D.elev_max_m - D.elev_min_m; sMax = D.slope_max_deg;

      reduced = window.matchMedia &&
                window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      var W = host.clientWidth || 800, H = host.clientHeight || 460;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      } catch (e) { fail("This needs WebGL, which this browser is not offering."); return; }
      if (!renderer || !renderer.getContext()) {
        fail("This needs WebGL, which this browser is not offering."); return;
      }
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(W, H);
      host.appendChild(renderer.domElement);

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(46, W / H, 0.5, 2000);

      scene.add(new THREE.AmbientLight(0xffffff, 0.62));
      var sun = new THREE.DirectionalLight(0xfff4e2, 0.85);
      sun.position.set(-90, 130, 70);
      scene.add(sun);
      var rim = new THREE.DirectionalLight(0xbcd2e8, 0.3);
      rim.position.set(80, 40, -90);
      scene.add(rim);

      buildTerrain();
      buildSurveyor();
      buildSummit();
      placeStart();

      camera.position.copy(camPos);
      camera.lookAt(camAim);

      bindInput();
      ready = true;
      if (pending) { var q = pending; pending = null; q(); }
      window.addEventListener("resize", resize);
      renderer.render(scene, camera);
      host.classList.add("live");

      if (window.IntersectionObserver) {
        new IntersectionObserver(function (es) {
          if (es[0].isIntersecting) { if (!raf) raf = requestAnimationFrame(frame); }
          else { if (raf) cancelAnimationFrame(raf); raf = null; active = false; }
        }, { threshold: 0.05 }).observe(host);
      }
      raf = requestAnimationFrame(frame);
    }

    /* ------------------------------------------------------- terrain, skirt, slab */

    function buildTerrain() {
      var nv = NROW * NCOL;
      var pos = new Float32Array(nv * 3);
      var col = new Float32Array(nv * 3);
      var c = new THREE.Color();

      for (var j = 0; j < NROW; j++) {
        for (var i = 0; i < NCOL; i++) {
          var k = j * NCOL + i, m = maskB[k];
          pos[k * 3] = wx(i);
          pos[k * 3 + 1] = m ? yAt(i, j) : 0;
          pos[k * 3 + 2] = wz(j);
          /* Ramp between band colours rather than stepping, so the slope of the risk is
             readable as well as the class. The class is what the rules use. */
          var p = probB[k] / 255, b = bandIndex(p);
          var lo = b === 0 ? 0 : BANDS[b - 1][0], hi = Math.min(BANDS[b][0], 1);
          var t = hi > lo ? (p - lo) / (hi - lo) : 0;
          var c0 = new THREE.Color(BANDS[b][1]);
          var c1 = new THREE.Color(BANDS[Math.min(4, b + 1)][1]);
          c.copy(c0).lerp(c1, Math.max(0, Math.min(1, t)) * 0.55);
          col[k * 3] = c.r; col[k * 3 + 1] = c.g; col[k * 3 + 2] = c.b;
        }
      }

      /* A face exists only where all four corners are inside the district, which is what makes
         the object the shape of Rangamati and not the shape of its bounding box. */
      var idx = [], edges = {};
      function edge(a, b) {
        var key = a < b ? a + ":" + b : b + ":" + a;
        edges[key] = (edges[key] || 0) + 1;
      }
      for (var jj = 0; jj < NROW - 1; jj++) {
        for (var ii = 0; ii < NCOL - 1; ii++) {
          var a = jj * NCOL + ii, b = a + 1, d = a + NCOL, e = d + 1;
          if (maskB[a] !== 2 || maskB[b] !== 2 || maskB[d] !== 2 || maskB[e] !== 2) continue;
          idx.push(a, d, b, b, d, e);
          edge(a, d); edge(d, b); edge(b, a);
          edge(b, d); edge(d, e); edge(e, b);
        }
      }

      topGeom = new THREE.BufferGeometry();
      topGeom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      topCol = new THREE.BufferAttribute(col, 3);
      topGeom.setAttribute("color", topCol);
      topGeom.setIndex(idx);
      topGeom.computeVertexNormals();

      terrain = new THREE.Mesh(topGeom, new THREE.MeshLambertMaterial({
        vertexColors: true, side: THREE.DoubleSide
      }));
      scene.add(terrain);

      /* Any edge used by a single triangle is on the silhouette. Dropping a wall from each one
         and capping the bottom turns the surface into a solid you could hold. */
      var sp = [], bottom = -DEPTH;
      for (var key in edges) {
        if (edges[key] !== 1) continue;
        var pr = key.split(":"), u = +pr[0], v = +pr[1];
        var ux = pos[u * 3], uy = pos[u * 3 + 1], uz = pos[u * 3 + 2];
        var vx = pos[v * 3], vy = pos[v * 3 + 1], vz = pos[v * 3 + 2];
        sp.push(ux, uy, uz, vx, vy, vz, ux, bottom, uz);
        sp.push(vx, vy, vz, vx, bottom, vz, ux, bottom, uz);
      }
      var skirtGeom = new THREE.BufferGeometry();
      skirtGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(sp), 3));
      skirtGeom.computeVertexNormals();
      scene.add(new THREE.Mesh(skirtGeom, new THREE.MeshLambertMaterial({
        color: 0x6B6156, side: THREE.DoubleSide, flatShading: true
      })));

      var bp = new Float32Array(idx.length * 3);
      for (var t2 = 0; t2 < idx.length; t2 += 3) {
        var order = [idx[t2], idx[t2 + 2], idx[t2 + 1]];      // reversed, so it faces down
        for (var q = 0; q < 3; q++) {
          var vtx = order[q];
          bp[(t2 + q) * 3] = pos[vtx * 3];
          bp[(t2 + q) * 3 + 1] = bottom;
          bp[(t2 + q) * 3 + 2] = pos[vtx * 3 + 2];
        }
      }
      var baseGeom = new THREE.BufferGeometry();
      baseGeom.setAttribute("position", new THREE.BufferAttribute(bp, 3));
      baseGeom.computeVertexNormals();
      scene.add(new THREE.Mesh(baseGeom, new THREE.MeshLambertMaterial({ color: 0x4F4840 })));
    }

    /* ------------------------------------------------------------------ the surveyor */

    function buildSurveyor() {
      surveyor = new THREE.Group();
      var ink = new THREE.MeshLambertMaterial({ color: 0x2E3A42 });
      var skin = new THREE.MeshLambertMaterial({ color: 0xE8D9C3 });
      var hat = new THREE.MeshLambertMaterial({ color: 0xB5791E });

      var torso = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.52, 1.15, 12), ink);
      torso.position.y = 1.35;
      surveyor.add(torso);

      var head = new THREE.Mesh(new THREE.SphereGeometry(0.38, 14, 12), skin);
      head.position.y = 2.2;
      surveyor.add(head);

      var cap = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.48, 0.3, 14), hat);
      cap.position.y = 2.46;
      surveyor.add(cap);
      var brim = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.62, 0.06, 16), hat);
      brim.position.y = 2.32;
      surveyor.add(brim);

      /* Pivots at the hip, so the legs swing from the top rather than the middle. */
      legL = new THREE.Group(); legR = new THREE.Group();
      legL.position.set(-0.22, 0.78, 0); legR.position.set(0.22, 0.78, 0);
      var legGeom = new THREE.BoxGeometry(0.22, 0.78, 0.24);
      var lm = new THREE.Mesh(legGeom, ink); lm.position.y = -0.39; legL.add(lm);
      var rm = new THREE.Mesh(legGeom, ink); rm.position.y = -0.39; legR.add(rm);
      surveyor.add(legL); surveyor.add(legR);

      scene.add(surveyor);
    }

    /* The highest cell inside the district, flagged. Wandering is more interesting with one
       place worth reaching, and the obstacle between here and there is not invented: the ground
       around a summit is exactly the ground the model scores worst. */
    function buildSummit() {
      var best = -1, bi = 0, bj = 0;
      for (var j = 0; j < NROW; j++) {
        for (var i = 0; i < NCOL; i++) {
          var k = j * NCOL + i;
          if (maskB[k] !== 2 || elevB[k] <= best) continue;
          best = elevB[k]; bi = i; bj = j;
        }
      }
      summit = new THREE.Vector3(wx(bi), yAt(bi, bj), wz(bj));

      summitFlag = new THREE.Group();
      var pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 4.2, 6),
                                new THREE.MeshLambertMaterial({ color: 0x2E3A42 }));
      pole.position.y = 2.1;
      summitFlag.add(pole);
      summitCloth = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.05),
        new THREE.MeshLambertMaterial({ color: 0xB5791E, side: THREE.DoubleSide }));
      summitCloth.position.set(0.85, 3.6, 0);
      summitFlag.add(summitCloth);
      summitFlag.position.copy(summit);
      scene.add(summitFlag);
    }

    /* ------------------------------------------------------------------ sampling */

    function gridOf(x, z) {
      return {
        fi: x / KM_LON + (NCOL - 1) / 2,
        fj: -z / KM_LAT + (NROW - 1) / 2
      };
    }

    function bilinear(buf, x, z) {
      var g = gridOf(x, z);
      var i0 = Math.max(0, Math.min(NCOL - 1, Math.floor(g.fi)));
      var j0 = Math.max(0, Math.min(NROW - 1, Math.floor(g.fj)));
      var i1 = Math.min(NCOL - 1, i0 + 1), j1 = Math.min(NROW - 1, j0 + 1);
      var tx = Math.max(0, Math.min(1, g.fi - i0)), tz = Math.max(0, Math.min(1, g.fj - j0));
      /* A corner with no data would drag the interpolation toward zero, so fall back to the
         cell the sample is actually in rather than inventing a value for it. */
      var base = buf[j0 * NCOL + i0];
      function at(i, j) { var k = j * NCOL + i; return maskB[k] ? buf[k] : base; }
      var a = at(i0, j0) * (1 - tx) + at(i1, j0) * tx;
      var b = at(i0, j1) * (1 - tx) + at(i1, j1) * tx;
      return a * (1 - tz) + b * tz;
    }

    function groundY(x, z) {
      return ((eMin + (bilinear(elevB, x, z) / 255) * eSpan) / 1000) * VEX;
    }
    function probOf(x, z) { return bilinear(probB, x, z) / 255; }
    function slopeOf(x, z) { return (bilinear(slopeB, x, z) / 255) * sMax; }
    function elevOf(x, z) { return eMin + (bilinear(elevB, x, z) / 255) * eSpan; }

    function insideDistrict(x, z) {
      var g = gridOf(x, z);
      var i = Math.round(g.fi), j = Math.round(g.fj);
      if (i < 0 || j < 0 || i >= NCOL || j >= NROW) return false;
      return maskB[j * NCOL + i] === 2;
    }

    /* Where the ground falls away fastest, which is the direction a failure would travel. */
    function downhill(x, z, out) {
      var d = 1.2;
      out.set(groundY(x - d, z) - groundY(x + d, z), 0, groundY(x, z - d) - groundY(x, z + d));
      if (out.lengthSq() < 1e-6) out.set(0, 0, 1);
      return out.normalize();
    }

    /* ------------------------------------------------------------------ placement */

    function placeStart() {
      /* Start on the calmest ground the model can find near the middle of the district, so the
         first thing anyone does is walk out of safety rather than into it. */
      var best = null, bestScore = 1e9;
      for (var j = 0; j < NROW; j++) {
        for (var i = 0; i < NCOL; i++) {
          var k = j * NCOL + i;
          if (maskB[k] !== 2) continue;
          var p = probB[k] / 255;
          if (p > 0.45) continue;
          var dj = Math.abs(j - NROW / 2) / NROW, di = Math.abs(i - NCOL / 2) / NCOL;
          var score = p + di * 0.8 + dj * 0.8;
          if (score < bestScore) { bestScore = score; best = [i, j]; }
        }
      }
      if (!best) best = [Math.floor(NCOL / 2), Math.floor(NROW / 2)];
      surveyor.position.set(wx(best[0]), 0, wz(best[1]));
      surveyor.position.y = groundY(surveyor.position.x, surveyor.position.z);
      safeSpot = surveyor.position.clone();
    }

    /* ------------------------------------------------------------------ input */

    var MOVE = { ArrowUp: "f", ArrowDown: "b", ArrowLeft: "l", ArrowRight: "r",
                 w: "f", s: "b", a: "l", d: "r", W: "f", S: "b", A: "l", D: "r" };

    function bindInput() {
      window.addEventListener("keydown", function (e) {
        if (!active) return;
        var m = MOVE[e.key];
        if (m) { keys[m] = true; e.preventDefault(); }
        else if (e.key === "Escape") { clearInput(); active = false; say("Paused. Touch or click the terrain to continue."); }
      });
      window.addEventListener("keyup", function (e) {
        var m = MOVE[e.key];
        if (m) keys[m] = false;
      });
      /* Losing the window with a key held would otherwise leave the surveyor jogging forever. */
      window.addEventListener("blur", clearInput);
      document.addEventListener("visibilitychange", function () {
        if (document.hidden) clearInput();
      });
      if (window.IntersectionObserver) {
        new IntersectionObserver(function (entries) {
          if (!entries[0].isIntersecting) clearInput();
        }).observe(host);
      }

      gestureRing = document.createElement("div");
      gestureRing.className = "walk-gesture";
      gestureRing.hidden = true;
      gestureRing.setAttribute("aria-hidden", "true");
      gestureThumb = document.createElement("span");
      gestureRing.appendChild(gestureThumb);
      host.appendChild(gestureRing);
      var surface = renderer.domElement;
      surface.addEventListener("pointerdown", gestureOn);
      surface.addEventListener("pointermove", gestureMove);
      surface.addEventListener("pointerup", gestureOff);
      surface.addEventListener("pointercancel", gestureOff);
      surface.addEventListener("lostpointercapture", gestureOff);
      surface.addEventListener("contextmenu", function (e) { e.preventDefault(); });

      if (startBtn) startBtn.addEventListener("click", start);
      if (intro) intro.addEventListener("click", function (e) {
        if (e.target === intro) start();
      });

    }

    function gestureOn(e) {
      if (e.button !== 0 || gestureId !== null) return;
      e.preventDefault();
      if (!started) start();
      active = true;
      gestureId = e.pointerId;
      gestureOriginX = e.clientX; gestureOriginY = e.clientY;
      gestureX = 0; gestureY = 0;
      var rect = host.getBoundingClientRect();
      gestureRing.style.left = (e.clientX - rect.left) + "px";
      gestureRing.style.top = (e.clientY - rect.top) + "px";
      gestureThumb.style.transform = "translate(-50%, -50%)";
      gestureRing.hidden = false;
      renderer.domElement.setPointerCapture(e.pointerId);
    }
    function gestureMove(e) {
      if (e.pointerId !== gestureId) return;
      e.preventDefault();
      var dx = e.clientX - gestureOriginX, dy = e.clientY - gestureOriginY;
      var distance = Math.hypot(dx, dy), radius = 38, deadZone = 6;
      var strength = Math.min(1, Math.max(0, (distance - deadZone) / (radius - deadZone)));
      gestureX = distance ? dx / distance * strength : 0;
      gestureY = distance ? dy / distance * strength : 0;
      var limit = distance ? Math.min(radius, distance) / distance : 0;
      gestureThumb.style.transform = "translate(-50%, -50%) translate(" + (dx * limit) + "px, " + (dy * limit) + "px)";
    }
    function gestureOff(e) {
      if (e.pointerId === gestureId) clearGesture();
    }
    function clearGesture() {
      var id = gestureId;
      gestureId = null; gestureX = 0; gestureY = 0;
      if (gestureRing) gestureRing.hidden = true;
      if (id !== null && renderer.domElement.hasPointerCapture(id)) renderer.domElement.releasePointerCapture(id);
    }
    function clearInput() {
      keys = {};
      clearGesture();
    }

    function start() {
      started = true;
      active = true;
      if (intro) intro.hidden = true;
      if (!raf) raf = requestAnimationFrame(frame);
      say(window.matchMedia("(any-pointer: coarse)").matches ?
        "Touch the terrain and slide to walk. Hold to keep moving; lift to stop." :
        "Use WASD, arrow keys, or drag the terrain to walk.");
    }

    var sayT = 0;
    function say(msg, kind) {
      if (!banner) return;
      banner.textContent = msg;
      banner.className = bannerBase + (kind ? " " + kind : "");
      banner.hidden = false;
      sayT = 3.4;
    }

    /* ------------------------------------------------------------------ the fall */

    function collapse(x, z, p) {
      state = "falling";
      fallT = 0;
      falls++;
      downhill(x, z, gradeDir);
      scar(x, z);
      /* Hold the readout on the cell that failed. The body slides on past the district edge,
         and reporting the ground under a falling man is both wrong and uninteresting. */
      fellAt = { x: x, z: z, p: p, e: elevOf(x, z), s: slopeOf(x, z) };
      say("The slope gave way at " + Math.round(elevOf(x, z)) + " m, on ground the model scores " +
          p.toFixed(2) + " — " + NAMES[bandIndex(p)] + ".", "bad");
    }

    /* Push the failed cells down and dull their colour. The mesh keeps the record. */
    function scar(x, z) {
      var g = gridOf(x, z);
      var ci = Math.round(g.fi), cj = Math.round(g.fj), R = 2;
      var pos = topGeom.attributes.position.array, col = topCol.array;
      for (var j = cj - R; j <= cj + R; j++) {
        for (var i = ci - R; i <= ci + R; i++) {
          if (i < 0 || j < 0 || i >= NCOL || j >= NROW) continue;
          var dist = Math.hypot(i - ci, j - cj);
          if (dist > R) continue;
          var k = j * NCOL + i, f = 1 - dist / (R + 0.6);
          pos[k * 3 + 1] -= 2.2 * f;
          col[k * 3] = col[k * 3] * (1 - f * 0.7) + 0.22 * f * 0.7;
          col[k * 3 + 1] = col[k * 3 + 1] * (1 - f * 0.7) + 0.17 * f * 0.7;
          col[k * 3 + 2] = col[k * 3 + 2] * (1 - f * 0.7) + 0.14 * f * 0.7;
        }
      }
      topGeom.attributes.position.needsUpdate = true;
      topCol.needsUpdate = true;
      topGeom.computeVertexNormals();
    }

    function respawn() {
      state = "walk";
      fuse = 0;
      surveyor.position.copy(safeSpot);
      surveyor.rotation.set(0, facing, 0);
      surveyor.visible = true;
      say("Back on ground the model calls safe.");
    }

    /* ------------------------------------------------------------------ frame */

    var last = 0;

    function frame(t) {
      raf = requestAnimationFrame(frame);
      var dt = last ? Math.min(0.05, (t - last) / 1000) : 0.016;
      last = t;
      clock += dt;

      if (started && state === "walk") step(dt);
      else if (state === "falling") fall(dt);

      if (sayT > 0) { sayT -= dt; if (sayT <= 0 && banner) banner.hidden = true; }

      aimCamera(dt);
      renderer.render(scene, camera);
    }

    function step(dt) {
      var vx = 0, vz = 0;
      if (keys.f) vz -= 1;
      if (keys.b) vz += 1;
      if (keys.l) vx -= 1;
      if (keys.r) vx += 1;
      if (gestureX || gestureY) {
        /* A screen-right drag moves right on the visible terrain, even while the camera settles. */
        var backX = camera.position.x - camAim.x, backZ = camera.position.z - camAim.z;
        var backLength = Math.hypot(backX, backZ) || 1;
        backX /= backLength; backZ /= backLength;
        vx += backZ * gestureX + backX * gestureY;
        vz += -backX * gestureX + backZ * gestureY;
      }

      var moving = vx || vz;
      if (moving) grace = false;
      if (moving) {
        var len = Math.hypot(vx, vz);
        var speed = SPEED * Math.min(1, len);
        vx = (vx / len) * speed * dt; vz = (vz / len) * speed * dt;
        var nx = surveyor.position.x + vx, nz = surveyor.position.z + vz;
        if (insideDistrict(nx, nz)) {
          surveyor.position.x = nx; surveyor.position.z = nz;
          walked += Math.hypot(vx, vz);
        } else {
          /* The model has nothing to say outside the district, so neither does the game. */
          if (!edgeSaid) { say("Edge of the district. The model stops where the data stops."); edgeSaid = true; }
          if (insideDistrict(nx, surveyor.position.z)) surveyor.position.x = nx;
          else if (insideDistrict(surveyor.position.x, nz)) surveyor.position.z = nz;
        }
        facing = Math.atan2(vx, vz);
        phase += dt * 11;
      } else {
        phase += dt * 1.6;
      }

      var x = surveyor.position.x, z = surveyor.position.z;
      surveyor.position.y = groundY(x, z);
      surveyor.rotation.y += angleDelta(surveyor.rotation.y, facing) * Math.min(1, dt * 12);

      var swing = moving ? Math.sin(phase) * 0.7 : Math.sin(phase) * 0.06;
      legL.rotation.x = swing;
      legR.rotation.x = -swing;
      surveyor.position.y += moving ? Math.abs(Math.sin(phase)) * 0.09 : 0;

      var p = probOf(x, z);
      if (p >= 0.80 && grace) {
        /* Standing on it is survivable while the grace lasts; the ground still shakes, so the
           reader can see what they have been handed before they take a step off it. */
        shake = 0.34;
      } else if (p >= 0.80) {
        fuse += dt;
        shake = Math.min(1, fuse / FUSE);
        if (fuse >= FUSE) collapse(x, z, p);
      } else {
        fuse = Math.max(0, fuse - dt * 1.6);
        shake = p >= 0.60 ? 0.28 : 0;
        if (p < 0.55) safeSpot.copy(surveyor.position);
      }

      if (summitFlag) {
        summitFlag.rotation.y = clock * 0.6;
        if (!summitMet && Math.hypot(x - summit.x, z - summit.z) < 3.5) {
          summitMet = true;
          summitCloth.material.color.setHex(0x1D6B58);
          say("The high point of the district, " + Math.round(D.elev_max_m) +
              " m, reached in " + walked.toFixed(1) + " km and " + falls +
              (falls === 1 ? " fall." : " falls."));
        }
      }

      paint(x, z, p);
    }

    var edgeSaid = false;

    function fall(dt) {
      fallT += dt;
      /* Tip over across the slope, then ride it down and under. */
      var tip = Math.min(1, fallT / 0.45);
      surveyor.rotation.z = tip * 1.5;
      surveyor.position.x += gradeDir.x * dt * 9 * Math.min(1, fallT * 2);
      surveyor.position.z += gradeDir.z * dt * 9 * Math.min(1, fallT * 2);
      surveyor.position.y = groundY(surveyor.position.x, surveyor.position.z)
                            - Math.max(0, fallT - 0.5) * 7;
      surveyor.rotation.y += dt * 4;
      shake = Math.max(0, 0.9 - fallT);
      if (fallT > 1.9) respawn();
      if (hud && fellAt) paint(fellAt.x, fellAt.z, fellAt.p);
    }

    function angleDelta(a, b) {
      var d = (b - a) % (Math.PI * 2);
      if (d > Math.PI) d -= Math.PI * 2;
      if (d < -Math.PI) d += Math.PI * 2;
      return d;
    }

    /* ------------------------------------------------------------------ camera, hud */

    function aimCamera(dt) {
      if (!started) {
        /* Near enough to a plan view to read as a map, turning slowly for the parallax that
           tells you it is a solid. The district is 211 units long in a viewport that is wider
           than it is tall, so the framing is set by its length, not by its bounding sphere.
           On a wide canvas the whole thing is nudged right, out from under the intro card. */
        var a = clock * 0.12;
        var off = camera.aspect > 1.4 ? -46 : 0;
        camTarget.set(off + Math.sin(a) * 55, 262, Math.cos(a) * 55);
        camLook.set(off, 0, 0);
      } else {
        camTarget.set(surveyor.position.x + 13, surveyor.position.y + 15,
                      surveyor.position.z + 20);
        camLook.set(surveyor.position.x, surveyor.position.y + 1.5, surveyor.position.z);
      }
      var k = Math.min(1, dt * (started ? 3.2 : 1.6));
      camPos.lerp(camTarget, k);
      camAim.lerp(camLook, Math.min(1, dt * 6));
      camera.position.copy(camPos);
      if (shake > 0 && !reduced) {
        var s = shake * 0.55;
        camera.position.x += (Math.random() - 0.5) * s;
        camera.position.y += (Math.random() - 0.5) * s;
      }
      camera.lookAt(camAim);
    }

    var lastBand = -1;

    /* One cell: a micro-label over the value, with the unit set quieter than the number and
       an optional dot carrying the band's own colour. The markup is class-based rather than
       tag-based so both host pages can style it with one block each. */
    function cell(label, value, unit, dot) {
      return '<span class="hc"><i>' + label + '</i><b>' +
             (dot ? '<u style="background:' + dot + '"></u>' : '') + value +
             (unit ? '<span class="hu">' + unit + '</span>' : '') + '</b></span>';
    }

    function paint(x, z, p) {
      if (!hud) return;
      var b = bandIndex(p);
      if (b !== lastBand) {
        hud.className = hudBase + " b" + b;
        lastBand = b;
      }
      hud.innerHTML =
        cell("elevation", Math.round(elevOf(x, z)), "m") +
        cell("slope", slopeOf(x, z).toFixed(1) + "\u00b0") +
        cell("susceptibility", p.toFixed(2), NAMES[b], BAND_HEX[b]) +
        cell("walked", walked.toFixed(1), "km") +
        cell("falls", falls);
    }

    function resize() {
      if (!renderer) return;
      var w = host.clientWidth, h = host.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    /* -------------------------------------------------------------- the public handle
       The portfolio has one of these on the page and never speaks to it again. The prediction
       app does: it owns a map and a dropdown, and when either of them names a place, the
       surveyor should already be standing there by the time the tab is opened. */

    function toWorld(lat, lon) {
      var fj = (lat - D.lat0) / D.step_deg;
      var fi = (lon - D.lon0) / D.step_deg;
      return { x: (fi - (NCOL - 1) / 2) * KM_LON, z: -(fj - (NROW - 1) / 2) * KM_LAT };
    }

    function latOf(z) { return D.lat0 + (-z / KM_LAT + (NROW - 1) / 2) * D.step_deg; }
    function lonOf(x) { return D.lon0 + (x / KM_LON + (NCOL - 1) / 2) * D.step_deg; }

    /* Put him down, on his feet, with the camera already in place. Sailing across the district
       to get there would be a nice shot and the wrong answer: the reader asked to be somewhere,
       not to watch a flight. */
    function land(x, z, note, kind) {
      surveyor.position.set(x, groundY(x, z), z);
      surveyor.rotation.set(0, facing, 0);
      surveyor.visible = true;
      safeSpot.copy(surveyor.position);
      state = "walk";
      fuse = 0; fallT = 0; shake = 0; fellAt = null;
      started = true; active = true;
      /* He did not choose to stand here, the page put him here, and a quarter of the district
         is ground that fails. So the countdown does not start until the reader moves him: the
         warning is visible, the consequence waits for a decision. */
      grace = true;
      if (intro) intro.hidden = true;
      camPos.set(x + 13, surveyor.position.y + 15, z + 20);
      camAim.set(x, surveyor.position.y + 1.5, z);
      camera.position.copy(camPos);
      camera.lookAt(camAim);
      var p = probOf(x, z);
      paint(x, z, p);
      if (note) {
        say(note + (placed ? "" : (window.matchMedia("(any-pointer: coarse)").matches ?
            "  Touch the terrain and slide to walk from here." : "  Arrow keys, WASD, or drag to walk from here.")),
            kind || (p >= 0.80 ? "bad" : ""));
      }
      placed = true;
      if (!raf) raf = requestAnimationFrame(frame);
      if (onArrive) {
        onArrive({ lat: latOf(z), lon: lonOf(x), prob: p, elev: elevOf(x, z),
                   slope: slopeOf(x, z), band: NAMES[bandIndex(p)] });
      }
      return true;
    }

    function moveTo(lat, lon, note) {
      if (!ready) { pending = function () { moveTo(lat, lon, note); }; return true; }
      var w = toWorld(lat, lon);
      if (!insideDistrict(w.x, w.z)) {
        say("That point is outside the district. The model has nothing to say about it, so the " +
            "surveyor stays where he is.", "bad");
        return false;
      }
      return land(w.x, w.z, note);
    }

    /* An inventory row carries factor ratings and no coordinates, so there is no such thing as
       the place it came from. The nearest honest answer is ground inside the district that falls
       in the same elevation and slope classes the row does, and among those, the cell whose score
       is the median, so what you get is a representative example rather than a flattering one. */
    function placeByClass(elevRating, slopeRating, note) {
      if (!ready) {
        pending = function () { placeByClass(elevRating, slopeRating, note); };
        return { queued: true };
      }
      var R = D.ratings, B = D.breaks;
      if (!R || !B) return null;
      var ei = R.ELEVATION.indexOf(elevRating), si = R.SLOPE.indexOf(slopeRating);
      if (ei < 0 || si < 0) return null;
      var eLo = ei === 0 ? -Infinity : B.ELEVATION[ei - 1];
      var eHi = ei >= B.ELEVATION.length ? Infinity : B.ELEVATION[ei];
      var sLo = si === 0 ? -Infinity : B.SLOPE[si - 1];
      var sHi = si >= B.SLOPE.length ? Infinity : B.SLOPE[si];

      var hits = [];
      for (var j = 0; j < NROW; j++) {
        for (var i = 0; i < NCOL; i++) {
          var k = j * NCOL + i;
          if (maskB[k] !== 2) continue;
          var e = eMin + (elevB[k] / 255) * eSpan;
          var sl = (slopeB[k] / 255) * sMax;
          if (e < eLo || e >= eHi || sl < sLo || sl >= sHi) continue;
          hits.push({ i: i, j: j, p: probB[k] });
        }
      }
      if (!hits.length) return { matches: 0 };
      hits.sort(function (a, b) { return a.p - b.p; });
      var pick = hits[Math.floor(hits.length / 2)];
      land(wx(pick.i), wz(pick.j), note);
      return {
        matches: hits.length,
        lat: +(D.lat0 + pick.j * D.step_deg).toFixed(4),
        lon: +(D.lon0 + pick.i * D.step_deg).toFixed(4),
        elev_lo: eLo === -Infinity ? null : eLo, elev_hi: eHi === Infinity ? null : eHi,
        slope_lo: sLo === -Infinity ? null : sLo, slope_hi: sHi === Infinity ? null : sHi
      };
    }

    return {
      moveTo: moveTo,
      placeByClass: placeByClass,
      resize: resize,
      /* The pane this lives in is usually hidden until a tab is opened, so the observer that
         would normally start it never fires. This lets the host say "now". */
      wake: function () { boot(); resize(); }
    };
  }

  var API = { init: init, current: null };

  /* A page that wants nothing but the default walk can say so in markup. */
  function auto() {
    var hosts = document.querySelectorAll("[data-walk]");
    for (var i = 0; i < hosts.length; i++) {
      var h = hosts[i], d = h.dataset;
      API.current = init({ host: h, data: d.walk, hud: d.walkHud, banner: d.walkBanner,
                           intro: d.walkIntro, start: d.walkStart });
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", auto);
  } else {
    auto();
  }

  return API;
})();
