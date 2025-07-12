// ==UserScript==
// @name         Change Dependency Viewer
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  Inject dependencies section in Gerrit UI
// @author       You
// @match        https://review.opendev.org/c/*
// @grant        none
// ==/UserScript==

// 🔧 Ce script est destiné uniquement aux tests de développement local.
// Ne pas déployer en production.

(function () {
    'use strict';
  
    function waitForGerritAndInject() {
      const interval = setInterval(() => {
        const app = document.querySelector('gr-app');
        if (!app?.shadowRoot) return;
  
        const appEl = app.shadowRoot.querySelector('gr-app-element');
        if (!appEl?.shadowRoot) return;
  
        const grChangeView = appEl.shadowRoot.querySelector('main > gr-change-view');
        if (!grChangeView?.shadowRoot) return;
  
        const metadata = grChangeView.shadowRoot.querySelector('#metadata');
        if (!metadata?.shadowRoot) return;
  
        const sections = metadata.shadowRoot.querySelectorAll('div > section');
        const targetSection = sections[5]; // 6e section
  
        if (!targetSection) return;
  
        clearInterval(interval);
  
        // ✅ Créer une nouvelle section
        const section = document.createElement('section');
        section.className = 'dependances';
        section.innerHTML = `
          <style>
            section.dependances {
              display: flex;
              align-items: center;
              padding: var(--spacing-m) 0;
              border-top: 1px solid var(--border-color);
            }
            .title {
              color: var(--deemphasized-text-color);
              min-width: 150px;
            }
          </style>
          <span class="title">
            <gr-tooltip-content has-tooltip title="Analysis of change dependencies">
              Dependencies
            </gr-tooltip-content>
          </span>
          <span class="value value-dependences">
            <gr-icon icon="schedule" filled></gr-icon>
          </span>
        `;
  
        targetSection.insertAdjacentElement('afterend', section);
  
        const valueSpan = section.querySelector('.value-dependences');
        const changeNumber = window.location.pathname.match(/\/\+\/(\d+)/)?.[1];
        if (!changeNumber) return;
  
        const payload = { change_number: parseInt(changeNumber) };
  
        // 🔁 Charger depuis ton API
        fetch('http://127.0.0.1:8000/generate-metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
    .then(resp => {
      if (!resp.ok) throw new Error(`Erreur HTTP: ${resp.status}`);
      return resp.json();
    })
    .then(pairs => {
      valueSpan.innerHTML = '';
      if (Array.isArray(pairs) && pairs.length > 0) {
        pairs.slice(0, 5).forEach(pairNumber => {
          const link = document.createElement('a');
          link.href = `https://review.opendev.org/c/${pairNumber}`;
          link.innerText = `#${pairNumber}`;
          link.style.cssText = `
            display: inline-block;
            margin-right: 8px;
            padding: 1px 8px;
            background: rgb(32, 33, 36);
            color: #ccc;
            border: 1px solid #555;
            border-radius: 9999px;
            text-decoration: none;
            font-size: 0.85rem;
          `;
          valueSpan.appendChild(link);
        });
      } else {
        valueSpan.innerHTML = `No dependencies for this change.`;
      }
    })
    .catch(err => {
      console.error('❌ Erreur de l’analyse :', err);
      valueSpan.innerHTML = `<gr-icon icon="error" filled></gr-icon>`;
    });
  
      }, 500);
    }
  
    waitForGerritAndInject();
  })();
  