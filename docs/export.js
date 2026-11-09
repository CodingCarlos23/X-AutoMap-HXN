/* HXN AutoMap docs — diagram PNG export + whole-page PDF export.
 *
 * Include after the Mermaid script on any docs page:
 *   <script src="export.js"></script>
 *
 * - Adds an "Export PNG" button to every .diagram-wrap (filename taken from the card's <h2>).
 * - Adds an "Export page as PDF" bar at the bottom of <main>; uses the browser's
 *   print dialog ("Save as PDF") with a print stylesheet so text stays selectable
 *   and diagrams stay vector.
 */
(function () {
  'use strict';

  const PNG_SCALE = 2; // 2x for crisp output on slides / papers

  // Styles that live in the page CSS (scoped under .mermaid) don't travel with a
  // detached SVG, so re-apply them inside the exported image.
  const SVG_EXPORT_CSS = `
    .node rect, .node polygon, .node path, .node circle, .node ellipse {
      filter: drop-shadow(0 2px 3px rgba(16, 37, 66, 0.14));
    }
    .cluster rect { filter: none; }
    .cluster-label, .cluster-label span { font-weight: 700; font-size: 11.5px; letter-spacing: 0.09em; }
    .nodeLabel b { font-size: 14px; }
    .nodeLabel small { display: inline-block; margin-top: 2px; font-size: 11.5px; opacity: 0.78;
                       font-family: "Roboto Mono", "Courier New", monospace; }
    .edgeLabel, .edgeLabel p { font-size: 12px; font-weight: 600; }
  `;

  const UI_CSS = `
    .diagram-wrap { position: relative; }
    .export-btn {
      position: absolute; top: 0.6rem; right: 0.6rem; z-index: 2;
      display: inline-flex; align-items: center; gap: 0.35rem;
      font: 600 0.75rem/1 "Public Sans", system-ui, sans-serif;
      color: #1a4480; background: #ffffff; border: 1px solid #c9d3df; border-radius: 4px;
      padding: 0.4rem 0.65rem; cursor: pointer; box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      transition: background 0.15s, border-color 0.15s;
    }
    .export-btn:hover { background: #eff6fb; border-color: #005ea2; }
    .export-btn:disabled { opacity: 0.6; cursor: progress; }
    .export-btn svg { width: 13px; height: 13px; }
    .export-all-bar {
      display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
      background: #ffffff; border-radius: 4px; border-left: 4px solid #1a4480;
      padding: 1.1rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
      margin-top: 2rem;
    }
    .export-all-bar p { margin: 0; font-size: 0.9rem; color: #565c65; }
    .export-all-bar .export-btn {
      position: static; font-size: 0.875rem; padding: 0.6rem 1.1rem;
      color: #ffffff; background: #1a4480; border-color: #1a4480;
    }
    .export-all-bar .export-btn:hover { background: #005ea2; border-color: #005ea2; }

    @media print {
      @page { size: letter; margin: 12mm; }
      html, body { background: #ffffff !important; }
      * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .site-nav, .export-btn, .export-all-bar { display: none !important; }
      .site-header { padding: 1rem 0 0.9rem; }
      main { max-width: none; margin: 1rem 0 0; padding: 0; }
      .card { box-shadow: none; border: 1px solid #e6e6e6; padding: 1.1rem 1.3rem; margin-bottom: 1rem; }
      .card h2 { break-after: avoid; }
      .card:has(.diagram-wrap) { break-inside: avoid; }
      .file-table { break-inside: avoid; }
      .diagram-wrap { break-inside: avoid; overflow: visible; padding: 1rem 0.5rem; }
      .diagram-wrap svg { max-height: 7in; width: auto; }
      .legend, .step-list li, .output-card, .model-card, tr { break-inside: avoid; }
      footer { margin-top: 1rem; }
    }
  `;

  const DOWNLOAD_ICON =
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M8 2v8M4.5 6.5 8 10l3.5-3.5M2.5 13.5h11"/></svg>';

  function slug(text) {
    return (text || 'diagram').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  }

  function pageStem() {
    const file = location.pathname.split('/').pop() || 'index.html';
    return 'hxn-' + file.replace(/\.html?$/, '');
  }

  /** Render a Mermaid <svg> element to a PNG Blob. */
  async function svgToPngBlob(svg, scale = PNG_SCALE) {
    const vb = svg.viewBox && svg.viewBox.baseVal;
    const rect = svg.getBoundingClientRect();
    const w = Math.ceil(vb && vb.width ? vb.width : rect.width);
    const h = Math.ceil(vb && vb.height ? vb.height : rect.height);

    const clone = svg.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    clone.setAttribute('width', w);
    clone.setAttribute('height', h);
    clone.style.maxWidth = 'none';
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = SVG_EXPORT_CSS;
    clone.insertBefore(style, clone.firstChild);

    const xml = new XMLSerializer().serializeToString(clone);
    const img = new Image();
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);
    await img.decode();

    const pad = 24;
    const canvas = document.createElement('canvas');
    canvas.width = (w + pad * 2) * scale;
    canvas.height = (h + pad * 2) * scale;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, pad * scale, pad * scale, w * scale, h * scale);

    return new Promise((resolve, reject) =>
      canvas.toBlob(b => (b ? resolve(b) : reject(new Error('PNG encode failed'))), 'image/png'));
  }

  function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function addDiagramButtons() {
    const used = {};
    document.querySelectorAll('.diagram-wrap').forEach(wrap => {
      const card = wrap.closest('.card');
      const title = card && card.querySelector('h2') ? card.querySelector('h2').textContent : 'diagram';
      let name = pageStem() + '-' + slug(title);
      used[name] = (used[name] || 0) + 1;
      if (used[name] > 1) name += '-' + used[name];

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'export-btn';
      btn.title = 'Download this diagram as a PNG image';
      btn.innerHTML = DOWNLOAD_ICON + '<span>Export PNG</span>';
      btn.addEventListener('click', async () => {
        const svg = wrap.querySelector('.mermaid svg');
        if (!svg) { alert('Diagram has not finished rendering yet.'); return; }
        const label = btn.querySelector('span');
        btn.disabled = true;
        label.textContent = 'Exporting…';
        try {
          download(await svgToPngBlob(svg), name + '.png');
        } catch (err) {
          console.error(err);
          alert('PNG export failed: ' + err.message);
        } finally {
          btn.disabled = false;
          label.textContent = 'Export PNG';
        }
      });
      wrap.appendChild(btn);
    });
  }

  function addExportAllBar() {
    const main = document.querySelector('main');
    if (!main) return;
    const bar = document.createElement('div');
    bar.className = 'export-all-bar';
    bar.innerHTML =
      '<p><strong>Export all</strong> — save this entire page, including every diagram, as a PDF. ' +
      'Choose <em>Save as PDF</em> as the destination in the print dialog.</p>';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'export-btn';
    btn.innerHTML = DOWNLOAD_ICON + '<span>Export page as PDF</span>';
    btn.addEventListener('click', () => {
      const prev = document.title;
      document.title = pageStem(); // becomes the default PDF filename
      window.addEventListener('afterprint', () => { document.title = prev; }, { once: true });
      window.print();
    });
    bar.appendChild(btn);
    main.appendChild(bar);
  }

  function init() {
    const style = document.createElement('style');
    style.textContent = UI_CSS;
    document.head.appendChild(style);
    addDiagramButtons();
    addExportAllBar();
  }

  window.HXNExport = { svgToPngBlob };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
