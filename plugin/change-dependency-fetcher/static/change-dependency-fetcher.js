Gerrit.install(plugin => {
  const createDependenciesSection = () => {
    // Trouver la section cible
    const targetSection = document.querySelector("#pg-app")
      ?.shadowRoot?.querySelector("#app-element")
      ?.shadowRoot?.querySelector("main > gr-change-view")
      ?.shadowRoot?.querySelector("#metadata")
      ?.shadowRoot?.querySelector("div > section:nth-child(6)");

    if (!targetSection) {
      setTimeout(createDependenciesSection, 500);
      return;
    }

    // Créer la section dépendances
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

    // Insérer après la section topic
    targetSection.insertAdjacentElement('afterend', section);

    // Récupérer les données
    const updateDependencies = async () => {
      const url = `http://127.0.0.1:8000/generate-metrics`
      const valueSpan = section.querySelector('.value-dependences');
      // const changeId = targetSection.closest('gr-change-metadata')?.change?._number;
      
      const getChange_number = () => {
        // Example URL format: /c/MyProject/+/12345
        const match = window.location.pathname.match(/\/\+\/(\d+)/);
        return match ? match[1] : null;
      }
      const change_number = getChange_number();

      if (!change_number) return;
      console.log('✍🏻 change ID: ', change_number);
      const payload = { change_number:  change_number};

      try {
        const icon = document.createElement('gr-icon');
        icon.setAttribute('icon', 'schedule');
        icon.setAttribute('filled', '');
        valueSpan.appendChild(icon);
        valueSpan.innerHTML = `<gr-icon icon="schedule" filled></gr-icon>`;
 
        // const response = await fetch(`/changes/${changeId}`);
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'  // <-- nécessaire
          },
          body: JSON.stringify(payload),
        });
        
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const pairs = await response.json();

        console.log('Change dependencies:', pairs);

        valueSpan.innerHTML = ''; // Efface l'icône de chargement
      // Ajouter dynamiquement un badge-lien pour chaque dépendance
      pairs.slice(0, 6).forEach(pairNumber => {
        const link = document.createElement('a');
        link.href = `https://review.opendev.org/c/${pairNumber}`;
        link.className = 'predicted-dependency-link';
        link.innerText = `${pairNumber}`;
        link.style.cssText = `
          display: inline-block;
          margin-right: 8px;
          padding: 1px 8px;
          // background: #e0ecff;
          // color: #1a73e8;
          background: rgb(32, 33, 36);       /* fond noir */
          color: var(--deemphasized-text-color);
          border: 0.5px solid var(--deemphasized-text-color);          
          border-radius: 9999px;
          text-decoration: none;
          font-size: 0.85rem;
          font-weight: 500;
        `;
        valueSpan.appendChild(link);
      });
        // valueSpan.innerHTML = `
        //     ${pairs[1]} 
        // `;
      } catch (error) {
        valueSpan.innerHTML = `  
            <gr-icon icon="error" filled></gr-icon>
        `;
        console.error('Erreur analyse:', error);
      }
    };
    updateDependencies();
  };

  // Démarrer avec un délai
  setTimeout(createDependenciesSection, 100);
});

