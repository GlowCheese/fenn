/* Fenn Dashboard — client-side interactivity */

(function () {
  "use strict";

  // ── Log viewer filters ─────────────────────────────────────────────────── //

  const levelState = { info: true, warning: true, exception: true };
  const kindState  = { system: true, user: true, print: true };

  function applyFilters() {
    const query = (document.getElementById("log-search")?.value || "").toLowerCase().trim();

    document.querySelectorAll(".log-entry").forEach((row) => {
      const level = row.dataset.level;
      const kind  = row.dataset.kind;
      const msg   = row.querySelector(".log-msg")?.textContent || "";

      const levelOk = levelState[level] !== false;
      const kindOk  = kindState[kind]   !== false;
      const searchOk = !query || msg.toLowerCase().includes(query);

      if (levelOk && kindOk && searchOk) {
        row.classList.remove("hidden", "search-hidden");
      } else {
        row.classList.add("hidden");
      }
    });

    // Highlight search matches
    if (query) {
      document.querySelectorAll(".log-entry:not(.hidden) .log-msg").forEach((el) => {
        const text = el.textContent;
        el.innerHTML = text.replace(
          new RegExp(escapeRegex(query), "gi"),
          (m) => `<span class="highlight">${m}</span>`
        );
      });
    } else {
      document.querySelectorAll(".log-msg .highlight").forEach((el) => {
        el.outerHTML = el.textContent;
      });
    }

    updateCount();
  }

  function updateCount() {
    const total   = document.querySelectorAll(".log-entry").length;
    const visible = document.querySelectorAll(".log-entry:not(.hidden)").length;
    const el = document.getElementById("log-visible-count");
    if (el) el.textContent = visible === total ? `${total} entries` : `${visible} / ${total} entries`;
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // Level filter buttons
  document.querySelectorAll("[data-filter-level]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const lvl = btn.dataset.filterLevel;
      if (lvl === "all") {
        Object.keys(levelState).forEach((k) => (levelState[k] = true));
        document.querySelectorAll("[data-filter-level]").forEach((b) => {
          b.classList.remove("active-info", "active-warning", "active-exception");
        });
        btn.classList.add("active-all");
      } else {
        levelState[lvl] = !levelState[lvl];
        document.querySelector("[data-filter-level='all']")?.classList.remove("active-all");
        btn.classList.toggle(`active-${lvl}`, levelState[lvl]);
      }
      applyFilters();
    });
  });

  // Kind filter buttons
  document.querySelectorAll("[data-filter-kind]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = btn.dataset.filterKind;
      if (k === "all") {
        Object.keys(kindState).forEach((key) => (kindState[key] = true));
        document.querySelectorAll("[data-filter-kind]").forEach((b) => {
          b.classList.remove("active-system", "active-user", "active-print");
        });
        btn.classList.add("active-all");
      } else {
        kindState[k] = !kindState[k];
        document.querySelector("[data-filter-kind='all']")?.classList.remove("active-all");
        btn.classList.toggle(`active-${k}`, kindState[k]);
      }
      applyFilters();
    });
  });

  // Search input
  const logSearch = document.getElementById("log-search");
  if (logSearch) {
    logSearch.addEventListener("input", applyFilters);
    logSearch.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { logSearch.value = ""; applyFilters(); }
    });
  }

  // ── Global session/project search (index page) ─────────────────────────── //

  const globalSearch = document.getElementById("global-search");
  if (globalSearch) {
    globalSearch.addEventListener("input", () => {
      const q = globalSearch.value.toLowerCase().trim();
      document.querySelectorAll("[data-searchable]").forEach((el) => {
        const text = el.dataset.searchable.toLowerCase();
        el.style.display = !q || text.includes(q) ? "" : "none";
      });
    });
  }

  // ── Collapsible sections ───────────────────────────────────────────────── //

  document.querySelectorAll(".collapsible-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.getElementById(btn.dataset.target);
      if (!target) return;
      const collapsed = target.classList.toggle("collapsed");
      const arrow = btn.querySelector(".arrow");
      if (arrow) arrow.textContent = collapsed ? "▶" : "▼";
      btn.setAttribute("aria-expanded", String(!collapsed));
    });
  });

  // ── Auto-refresh for running sessions ──────────────────────────────────── //

  const sessionStatus = document.getElementById("session-status");
  if (sessionStatus && sessionStatus.dataset.status === "running") {
    const project   = sessionStatus.dataset.project;
    const sessionId = sessionStatus.dataset.session;

    function refresh() {
      fetch(`/api/session/${encodeURIComponent(project)}/${encodeURIComponent(sessionId)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (!data) return;

          // Reload if status changed or entry count changed
          const currentCount = document.querySelectorAll(".log-entry").length;
          if (data.entry_count !== currentCount || data.status !== "running") {
            location.reload();
          }
        })
        .catch(() => {});
    }

    setInterval(refresh, 5000);
  }

  // ── Keyboard shortcuts ─────────────────────────────────────────────────── //

  document.addEventListener("keydown", (e) => {
    // Ctrl/Cmd + K → focus search
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      const s = document.getElementById("log-search") || document.getElementById("global-search");
      if (s) { s.focus(); s.select(); }
    }
    // Escape → blur search
    if (e.key === "Escape") {
      document.activeElement?.blur();
    }
  });

  // ── Init ───────────────────────────────────────────────────────────────── //

  updateCount();

})();
