(() => {
  const boxSelector = [
    '.hero-card',
    '.card',
    '.chart-panel',
    '.step',
    '.stat',
    '.threshold',
    '.matrix-cell',
    '.result-banner',
    '.chart-card'
  ].join(',');

  const animateBoxes = () => {
    const boxes = document.querySelectorAll(boxSelector);
    boxes.forEach((box, index) => {
      box.classList.add('anim-box');
      box.style.setProperty('--delay', `${Math.min(index * 55, 520)}ms`);
    });
  };

  const animateBars = () => {
    const bars = document.querySelectorAll('.animated-bar[data-width]');
    bars.forEach((bar, index) => {
      const targetWidth = bar.getAttribute('data-width') || '0%';
      window.setTimeout(() => {
        bar.style.width = targetWidth;
      }, 120 + Math.min(index * 90, 540));
    });
  };

  const wirePageTransitions = () => {
    const links = document.querySelectorAll('a[href]');
    links.forEach((link) => {
      link.addEventListener('click', (event) => {
        const href = link.getAttribute('href');
        if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) {
          return;
        }

        const url = new URL(href, window.location.href);
        const current = new URL(window.location.href);

        if (url.origin !== current.origin) {
          return;
        }

        event.preventDefault();
        document.body.classList.add('page-transition-out');
        window.setTimeout(() => {
          window.location.href = url.href;
        }, 280);
      });
    });
  };

  document.addEventListener('DOMContentLoaded', () => {
    animateBoxes();
    animateBars();
    wirePageTransitions();
  });
})();
