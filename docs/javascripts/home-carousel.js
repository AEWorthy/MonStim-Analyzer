(() => {
  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");
  const ROTATE_MS = 7000;

  function initialiseCarousel(carousel) {
    if (carousel.dataset.initialised === "true") return;
    carousel.dataset.initialised = "true";
    const slides = Array.from(carousel.querySelectorAll(".demo-carousel__slide"));
    const previous = carousel.querySelector("[data-demo-carousel-previous]");
    const next = carousel.querySelector("[data-demo-carousel-next]");
    const status = carousel.querySelector("[data-demo-carousel-status]");
    const autoplay = carousel.querySelector("[data-demo-carousel-autoplay]");
    const autoplayStatus = carousel.querySelector("[data-demo-carousel-autoplay-status]");
    const progress = carousel.querySelector("[data-demo-carousel-progress]");
    if (!slides.length || !previous || !next || !status || !autoplay || !autoplayStatus || !progress) return;

    let index = 0;
    let timer = null;
    const restartProgress = () => {
      progress.style.animation = "none";
      void progress.offsetWidth;
      progress.style.animation = "";
      autoplay.classList.remove("is-paused");
    };
    const show = (newIndex) => {
      index = (newIndex + slides.length) % slides.length;
      slides.forEach((slide, slideIndex) => {
        const active = slideIndex === index;
        slide.classList.toggle("is-active", active);
        slide.setAttribute("aria-hidden", String(!active));
      });
      status.textContent = `${index + 1} / ${slides.length}`;
    };
    const stop = () => {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
      if (!REDUCED_MOTION.matches) {
        autoplay.classList.add("is-paused");
        autoplayStatus.textContent = "Carousel rotation paused.";
      }
    };
    const start = () => {
      if (timer !== null) window.clearInterval(timer);
      timer = null;
      if (!REDUCED_MOTION.matches) {
        restartProgress();
        autoplayStatus.textContent = "Auto-advancing every 7 seconds.";
        timer = window.setInterval(() => {
          if (!carousel.isConnected) return stop();
          show(index + 1);
          restartProgress();
        }, ROTATE_MS);
      } else {
        autoplay.classList.add("is-paused");
        autoplayStatus.textContent = "Auto-rotation disabled by motion preference";
      }
    };

    carousel.classList.add("is-enhanced");
    show(0);
    previous.addEventListener("click", () => {
      show(index - 1);
      start();
    });
    next.addEventListener("click", () => {
      show(index + 1);
      start();
    });
    carousel.addEventListener("mouseenter", stop);
    carousel.addEventListener("mouseleave", start);
    carousel.addEventListener("focusin", stop);
    carousel.addEventListener("focusout", (event) => {
      if (!carousel.contains(event.relatedTarget)) start();
    });
    REDUCED_MOTION.addEventListener("change", start);
    start();
  }

  function initialise() {
    document.querySelectorAll("[data-demo-carousel]").forEach(initialiseCarousel);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initialise);
  } else {
    document.addEventListener("DOMContentLoaded", initialise);
  }
})();
