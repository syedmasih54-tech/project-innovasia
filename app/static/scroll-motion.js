/* One-time section reveals. Presentation and native scrolling stay unchanged. */
(() => {
  'use strict';

  if (!('IntersectionObserver' in window) || !Element.prototype.animate) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const activeAnimations = new Map();
  const seen = new WeakSet();
  let observer;

  function finishMotion() {
    activeAnimations.forEach(animation => animation.cancel());
    activeAnimations.clear();
  }

  function reveal(entries) {
    let stagger = 0;
    entries.forEach(entry => {
      if (!entry.isIntersecting || seen.has(entry.target)) return;
      const element = entry.target;
      seen.add(element);
      observer.unobserve(element);

      // Never move a focused control, restored content above the viewport,
      // or content while the user has requested reduced motion.
      if (reducedMotion.matches || document.hidden ||
          element.matches(':focus-within') || entry.boundingClientRect.top < 0) return;

      const animation = element.animate([
        { opacity: 0.45, translate: '0 18px' },
        { opacity: 1, translate: '0 0' }
      ], {
        duration: 540,
        delay: Math.min(stagger++ * 65, 195),
        easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
        fill: 'backwards'
      });
      // Individual translate preserves the existing hover and bank 3D transforms.
      activeAnimations.set(element, animation);
      const cleanup = () => {
        if (activeAnimations.get(element) === animation) activeAnimations.delete(element);
      };
      animation.onfinish = cleanup;
      animation.oncancel = cleanup;
    });
  }

  function start() {
    if (observer) return;
    observer = new IntersectionObserver(reveal, {
      threshold: 0,
      rootMargin: '0px 0px -32px 0px'
    });

    // Animate existing section blocks, splitting grid rows into a gentle stagger.
    // Do not select live table rows, metrics, graph nodes, or nested controls.
    const observeBlock = block => {
      if (block.classList.contains('grid')) {
        Array.from(block.children).forEach(child => observer.observe(child));
      } else {
        observer.observe(block);
      }
    };
    document.querySelectorAll('.tab-pane').forEach(pane => {
      Array.from(pane.children).forEach(section => {
        if (section.querySelector(':scope > .grid')) {
          Array.from(section.children).forEach(observeBlock);
        } else {
          observeBlock(section);
        }
      });
    });
  }

  // Keyboard interaction and preference changes immediately restore resting styles.
  document.addEventListener('focusin', event => {
    activeAnimations.forEach((animation, element) => {
      if (element.contains(event.target)) {
        animation.cancel();
        activeAnimations.delete(element);
      }
    });
  });
  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) finishMotion();
  });
  window.addEventListener('pagehide', finishMotion);
  window.addEventListener('beforeprint', finishMotion);

  // Wait for the existing intro to leave, so the first reveals are actually seen.
  const preloader = document.getElementById('streamshield-preloader');
  if (preloader && getComputedStyle(preloader).display !== 'none') {
    const introObserver = new MutationObserver(() => {
      if (getComputedStyle(preloader).display === 'none') {
        introObserver.disconnect();
        start();
      }
    });
    introObserver.observe(preloader, { attributes: true, attributeFilter: ['style', 'class'] });
  } else {
    start();
  }
})();
