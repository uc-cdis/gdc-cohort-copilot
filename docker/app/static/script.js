const GDC_API_URL = "https://api.gdc.cancer.gov/cases";
const FACETS = "project.program.name,project.project_id,disease_type,diagnoses.primary_diagnosis,primary_site,diagnoses.tissue_or_organ_of_origin,demographic.gender,demographic.race,demographic.ethnicity,diagnoses.age_at_diagnosis,demographic.vital_status,diagnoses.ajcc_clinical_stage,diagnoses.ajcc_pathologic_stage,diagnoses.uicc_clinical_stage,diagnoses.uicc_pathologic_stage,diagnoses.tumor_grade,diagnoses.morphology,diagnoses.year_of_diagnosis,diagnoses.site_of_resection_or_biopsy,diagnoses.sites_of_involvement,diagnoses.laterality,diagnoses.prior_malignancy,diagnoses.prior_treatment,diagnoses.synchronous_malignancy,diagnoses.progression_or_recurrence,diagnoses.residual_disease,diagnoses.child_pugh_classification,diagnoses.ishak_fibrosis_score,diagnoses.ann_arbor_clinical_stage,diagnoses.ann_arbor_pathologic_stage,diagnoses.cog_renal_stage,diagnoses.figo_stage,diagnoses.igcccg_stage,diagnoses.inss_stage,diagnoses.iss_stage,diagnoses.masaoka_stage,diagnoses.inpc_grade,diagnoses.who_cns_grade,diagnoses.cog_neuroblastoma_risk_group,diagnoses.cog_rhabdomyosarcoma_risk_group,diagnoses.eln_risk_classification,diagnoses.international_prognostic_index,diagnoses.wilms_tumor_histologic_subtype,diagnoses.weiss_assessment_score,diagnoses.best_overall_response,diagnoses.treatments.therapeutic_agents,diagnoses.treatments.treatment_intent_type,diagnoses.treatments.treatment_outcome,diagnoses.treatments.treatment_type,exposures.alcohol_history,exposures.alcohol_intensity,exposures.tobacco_smoking_status,exposures.cigarettes_per_day,exposures.pack_years_smoked,exposures.tobacco_smoking_onset_year,samples.tissue_type,samples.biospecimen_anatomic_site,samples.specimen_type,samples.preservation_method,samples.tumor_descriptor,samples.portions.analytes.aliquots.analyte_type,files.data_category,files.data_type,files.experimental_strategy,files.analysis.workflow_type,files.data_format,files.platform,files.access";

const BACKEND_NL_URL = "/api/cohort"; // Change this to your backend endpoint as needed

let filterData = {};
let mapping = {};
let tabOrder = [];
let tabCards = {};
let apiAggs = {};
let currentSelections = {
  checkboxes: {},
  ranges: {}
};
let latestFilterJson = ""
let lastNLJson = null;
let nlModalCollapsed = false;

// Dummy JSON for modal test
const DUMMY_JSON = {
  "op": "and",
  "content": [
    { "op": "in", "content": { "field": "cases.project.program.name", "value": ["FM"] } },
    { "op": "in", "content": { "field": "cases.disease_type", "value": ["adenomas and adenocarcinomas"] } },
    { "op": "in", "content": { "field": "cases.demographic.gender", "value": ["male"]}},
    { "op": "in", "content": { "field": "files.access", "value": ["open"] } },
  ]
};

function showLoader() {
  let loader = document.getElementById('spinner-overlay');
  if (!loader) {
    loader = document.createElement('div');
    loader.id = 'spinner-overlay';
    loader.style.position = 'fixed';
    loader.style.top = 0;
    loader.style.left = 0;
    loader.style.width = '100vw';
    loader.style.height = '100vh';
    loader.style.background = 'rgba(255,255,255,0.4)';
    loader.style.display = 'flex';
    loader.style.alignItems = 'center';
    loader.style.justifyContent = 'center';
    loader.style.zIndex = 9999;
    loader.innerHTML = `
      <div class="flex flex-col items-center">
        <svg class="animate-spin h-12 w-12 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
        </svg>
        <div class="mt-4 text-blue-600 text-lg font-semibold">Loading...</div>
      </div>
    `;
    document.body.appendChild(loader);
  } else {
    loader.style.display = 'flex';
  }
}
function hideLoader() {
  const loader = document.getElementById('spinner-overlay');
  if (loader) loader.style.display = 'none';
}

async function fetchApiCounts(filters = {}) {
  const params = new URLSearchParams();
  params.set('facets', FACETS);
  params.set('pretty', 'false');
  params.set('format', 'JSON');
  if (Object.keys(filters).length > 0) {
    params.set('filters', JSON.stringify(filters));
  }
  const url = `${GDC_API_URL}?${params.toString()}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return await resp.json();
}

function updateCaseCountBar(totalCases) {
  const el = document.getElementById('case-count-bar');
  if (typeof totalCases === "number" && !isNaN(totalCases)) {
    el.textContent = `Cases in cohort: ${totalCases.toLocaleString()}`;
  } else {
    el.textContent = "";
  }
}

function stripCasesPrefix(field) {
  if (field.startsWith("cases.") && !field.startsWith("cases.files.") && !field.startsWith("files.")) {
    return field.slice(6); // Remove "cases."
  }
  return field;
}


function restoreSelectionsToUI(mainContent) {
  // Restore checkboxes
  mainContent.querySelectorAll('input[type=checkbox]').forEach(cb => {
    const field = cb.getAttribute('data-field'); // always field path now!
    const val = cb.getAttribute('data-value');
    if (currentSelections.checkboxes[field] && currentSelections.checkboxes[field].has(val)) {
      cb.checked = true;
    }
  });
  // Restore range fields
  mainContent.querySelectorAll('.min-input').forEach((minInput) => {
    const card = minInput.closest('.bg-white');
    if (!card) return;
    const cardName = card.querySelector('.card-header-band').innerText.trim();
    const apiAggKey = mapping[cardName]; // remove strip func
    if (apiAggKey && currentSelections.ranges[apiAggKey]) {
      minInput.value = currentSelections.ranges[apiAggKey].min || "";
      const maxInput = card.querySelector('.max-input');
      if (maxInput) maxInput.value = currentSelections.ranges[apiAggKey].max || "";
    }
  });
}

function buildGdcFilters() {
  const content = [];
  for (const fieldPath in currentSelections.checkboxes) {
    const values = Array.from(currentSelections.checkboxes[fieldPath]);
    if (values.length > 0) {
      content.push({
        op: "in",
        content: {field: fieldPath, value: values}
      });
    }
  }
  for (const fieldPath in currentSelections.ranges) {
    const {min, max} = currentSelections.ranges[fieldPath];
    if (min || max) {
      const rangeContent = {field: fieldPath};
      if (min) rangeContent.gte = Number(min);
      if (max) rangeContent.lte = Number(max);
      content.push({op: "range", content: rangeContent});
    }
  }
  if (content.length === 0) return {};
  return {op: "and", content};
}

function updateJsonDropdown(json) {
    latestFilterJson = json; // store latest always
    // document.getElementById('dropdown-json-content').textContent = JSON.stringify(json, null, 2);
    document.getElementById('dropdown-json-content').textContent = json; // value vs text contenxt
  }

function showJsonDropdown() {
  document.getElementById('filter-json-dropdown').classList.remove('hidden');
  // Position just below the search bar (optionally adjust style.top)
}

function hideJsonDropdown() {
  document.getElementById('filter-json-dropdown').classList.add('hidden');
}

async function loadAllData() {
  showLoader();
  // 1. Field mappings
  const yamlText = await fetch('field_mappings.yaml').then(r => r.text());
  mapping = jsyaml.load(yamlText).keys;

  // 2. filterData.json (tabs/cards)
  filterData = await fetch('filterData.json').then(r => r.json());
  tabOrder = Object.keys(filterData);
  tabCards = {};
  for (const tab of tabOrder) tabCards[tab] = Object.keys(filterData[tab]);

  // 3. API aggregations
  const apiJson = await fetchApiCounts();
  apiAggs = apiJson.data.aggregations;
  const apiTotal = apiJson.data.pagination.total;
  updateCaseCountBar(typeof apiTotal === "number" ? apiTotal : null);
  // Clear selections
  currentSelections = { checkboxes: {}, ranges: {} };

  renderTabs();
  renderUI(tabOrder[0]);
  hideLoader();
}

function renderTabs() {
  const sidebar = document.getElementById('sidebar-tabs');
  sidebar.innerHTML = '<h2 class="text-lg font-semibold mb-4">Tabs</h2>';
  tabOrder.forEach((tab, idx) => {
    const btn = document.createElement('button');
    btn.id = `tab-btn-${tab}`;
    btn.innerText = tab;
    btn.className = 'w-full text-left px-4 py-2 rounded transition font-medium mb-1';
    if (idx === 0) btn.classList.add('bg-blue-100', 'text-blue-600');
    else btn.classList.add('text-gray-700');
    btn.onclick = () => {
      tabOrder.forEach(otherTab => {
        document.getElementById(`tab-btn-${otherTab}`).classList.remove('bg-blue-100', 'text-blue-600');
        document.getElementById(`tab-btn-${otherTab}`).classList.add('text-gray-700');
      });
      btn.classList.add('bg-blue-100', 'text-blue-600');
      renderUI(tab);
    };
    sidebar.appendChild(btn);
  });
}



// function renderUI(currentTab) {
//   const mainContent = document.getElementById('main-content');
//   mainContent.innerHTML = '';
//   const grid = document.createElement('div');
//   grid.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4';

//   for (const cardName of tabCards[currentTab]) {
//     const cardConfig = filterData[currentTab][cardName];
//     const apiAggKey = stripCasesPrefix(mapping[cardName]);
//     const card = document.createElement('div');
//     card.className = 'bg-white p-4 rounded shadow relative';
//     const header = document.createElement('div');
//     header.className = 'card-header-band';
//     header.innerText = cardName;
//     card.appendChild(header);

//     if (!apiAggKey) {
//       const dbg = document.createElement('div');
//       dbg.className = "card-debug";
//       dbg.innerText = `Not rendered: No mapping for card "${cardName}" in field_mappings.yaml.`;
//       card.appendChild(dbg);
//       grid.appendChild(card);
//       continue;
//     }

//     // ---- RANGE/RANGE+CHECKBOX CARDS ----
//     if (
//       typeof cardConfig === "object" &&
//       cardConfig !== null &&
//       !Array.isArray(cardConfig) &&
//       cardConfig.type
//     ) {
//       const rangeRow = document.createElement('div');
//       rangeRow.className = 'flex space-x-2 items-end mb-2';
//       const minGroup = document.createElement('div');
//       minGroup.innerHTML = `<label class="text-xs text-gray-500">${cardConfig.min_label || "Min"}</label>
//           <input type="number" placeholder="${cardConfig.min_label || "Min"}" class="min-input border rounded w-20 p-1 focus:ring focus:ring-blue-200" min="${cardConfig.min}" max="${cardConfig.max}">`;
//       rangeRow.appendChild(minGroup);
//       const dash = document.createElement('span');
//       dash.className = 'mx-1 text-gray-400';
//       dash.innerText = '—';
//       rangeRow.appendChild(dash);
//       const maxGroup = document.createElement('div');
//       maxGroup.innerHTML = `<label class="text-xs text-gray-500">${cardConfig.max_label || "Max"} </label>
//           <input type="number" placeholder="${cardConfig.max_label || "Max"}" class="max-input border rounded w-20 p-1 focus:ring focus:ring-blue-200" min="${cardConfig.min}" max="${cardConfig.max}">`;
//       rangeRow.appendChild(maxGroup);
//       if (cardConfig.unit) {
//         const unitSpan = document.createElement('span');
//         unitSpan.className = 'ml-2 text-xs text-gray-400';
//         unitSpan.innerText = cardConfig.unit;
//         rangeRow.appendChild(unitSpan);
//       }
//       card.appendChild(rangeRow);

//       // Add robust range input listeners
//       const minInput = rangeRow.querySelector('.min-input');
//       const maxInput = rangeRow.querySelector('.max-input');
//       [minInput, maxInput].forEach(input =>
//         input.addEventListener('input', () => {
//           const minVal = minInput.value;
//           const maxVal = maxInput.value;
//           if (minVal || maxVal) {
//             currentSelections.ranges[apiAggKey] = { min: minVal, max: maxVal };
//           } else {
//             delete currentSelections.ranges[apiAggKey];
//           }
//           onAnyFilterChange();
//         })
//       );

//       if (cardConfig.type === "range+checkboxes" && Array.isArray(cardConfig.popular)) {
//         const popDiv = document.createElement('div');
//         popDiv.className = 'popular-range-row flex flex-wrap gap-2 mt-2';

//         let buckets = [];
//         if (apiAggs[apiAggKey] && apiAggs[apiAggKey].buckets)
//           buckets = apiAggs[apiAggKey].buckets;

//         const cardKey = `${currentTab}-${cardName}`.replace(/[^a-zA-Z0-9_-]/g, '');

//         cardConfig.popular.forEach((pop, i) => {
//           let popCount = 0;
//           let popPct = 0;
//           if (buckets.length) {
//             popCount = buckets.filter(
//               b => b.key !== "_missing" && b.doc_count > 0 &&
//                 (Number(b.key) >= pop.min && Number(b.key) <= pop.max)
//             ).reduce((acc, b) => acc + b.doc_count, 0);
//             const total = buckets.filter(b => b.doc_count > 0 && b.key !== "_missing")
//                                 .reduce((acc, b) => acc + b.doc_count, 0);
//             popPct = total ? ((popCount / total) * 100).toFixed(1) : 0;
//           }
//           const label = document.createElement('label');
//           label.className = 'inline-flex items-center px-2 py-1 bg-gray-100 rounded cursor-pointer';
//           // label.innerHTML = `<input type='checkbox' data-field='${apiAggKey}' class='popular-range mr-2' name='${cardKey}-pop' data-min='${pop.min}' data-max='${pop.max}' id='${cardKey}-pop-${i}'>${pop.label}
//           label.innerHTML = `<input type='checkbox' data-field='${mapping[cardName]}' class='popular-range mr-2' name='${cardKey}-pop' data-min='${pop.min}' data-max='${pop.max}' id='${cardKey}-pop-${i}'>${pop.label}
//               <span class="checkbox-count">${popCount ? popCount.toLocaleString() : 0} (${popPct}%)</span>`;
//           popDiv.appendChild(label);
//         });
//         card.appendChild(popDiv);

//         // SYNC logic: listener for popular range checkboxes
//         const checkboxes = popDiv.querySelectorAll('input[type=checkbox]');
//         checkboxes.forEach(cb => {
//           cb.addEventListener('change', () => {
//             if (cb.checked) {
//               checkboxes.forEach(other => { if (other !== cb) other.checked = false; });
//               minInput.value = cb.dataset.min;
//               maxInput.value = cb.dataset.max;
//               currentSelections.ranges[apiAggKey] = { min: cb.dataset.min, max: cb.dataset.max };
//             } else {
//               if (!minInput.value && !maxInput.value) {
//                 delete currentSelections.ranges[apiAggKey];
//               }
//             }
//             onAnyFilterChange();
//           });
//         });
//       }
//       grid.appendChild(card);
//       continue;
//     }

//     // ---- STANDARD CHECKBOX CARDS ----
//     if (
//       Array.isArray(cardConfig) ||
//       cardConfig === null ||
//       (typeof cardConfig === "object" && cardConfig !== null && !cardConfig.type && Object.keys(cardConfig).length === 0)
//     ) {
//       let rendered = false;
//       if (apiAggs[apiAggKey] && apiAggs[apiAggKey].buckets) {
//         const buckets = apiAggs[apiAggKey].buckets.filter(
//           b => b.doc_count > 0 && b.key !== "_missing"
//         ).sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true, sensitivity: 'base' }));
//         if (buckets.length > 0) {
//           const total = buckets.reduce((a, b) => a + b.doc_count, 0);
//           buckets.slice(0, 6).forEach(bucket => {
//             const pct = ((bucket.doc_count / total) * 100).toFixed(1);
//             const row = document.createElement('div');
//             row.className = "checkbox-row";
//             // row.innerHTML = `
//             //   <label class="flex items-center checkbox-label">
//             //     <input type="checkbox" data-field="${apiAggKey}" data-value="${bucket.key}" class="mr-2">
//             //     ${bucket.key}
//             //   </label>
//             //   <span class="checkbox-count">${bucket.doc_count.toLocaleString()} (${pct}%)</span>
//             // `;
//             row.innerHTML = `
//               <label class="flex items-center checkbox-label">
//                 <input type="checkbox" data-field="${mapping[cardName]}" data-value="${bucket.key}" class="mr-2">
//                 ${bucket.key}
//               </label>
//               <span class="checkbox-count">${bucket.doc_count.toLocaleString()} (${pct}%)</span>
//             `;
//             card.appendChild(row);
//             const input = row.querySelector('input[type=checkbox]');
//             // Restore checked state
//             if (
//               currentSelections.checkboxes[apiAggKey] &&
//               currentSelections.checkboxes[apiAggKey].has(bucket.key)
//             ) {
//               input.checked = true;
//             }
//             // Robust state: update only that field/value!
//             input.addEventListener('change', () => {
//               if (!currentSelections.checkboxes[apiAggKey]) currentSelections.checkboxes[apiAggKey] = new Set();
//               if (input.checked) {
//                 currentSelections.checkboxes[apiAggKey].add(bucket.key);
//               } else {
//                 currentSelections.checkboxes[apiAggKey].delete(bucket.key);
//                 if (currentSelections.checkboxes[apiAggKey].size === 0) delete currentSelections.checkboxes[apiAggKey];
//               }
//               onAnyFilterChange();
//             });
//           });
//           if (buckets.length > 6) {
//             const id = `${cardName.replace(/[^a-zA-Z0-9_-]/g, '')}-more`;
//             const extraDiv = document.createElement('div');
//             extraDiv.id = id;
//             extraDiv.className = 'hidden';
//             buckets.slice(6).forEach(bucket => {
//               const pct = ((bucket.doc_count / total) * 100).toFixed(1);
//               const row = document.createElement('div');
//               row.className = "checkbox-row";
//               // row.innerHTML = `
//               //   <label class="flex items-center checkbox-label">
//               //     <input type="checkbox" data-field="${apiAggKey}" data-value="${bucket.key}" class="mr-2">
//               //     ${bucket.key}
//               //   </label>
//               //   <span class="checkbox-count">${bucket.doc_count.toLocaleString()} (${pct}%)</span>
//               // `;
//               row.innerHTML = `
//                 <label class="flex items-center checkbox-label">
//                   <input type="checkbox" data-field="${mapping[cardName]}" data-value="${bucket.key}" class="mr-2">
//                   ${bucket.key}
//                 </label>
//                 <span class="checkbox-count">${bucket.doc_count.toLocaleString()} (${pct}%)</span>
//               `;
//               extraDiv.appendChild(row);
//               const input = row.querySelector('input[type=checkbox]');
//               if (
//                 currentSelections.checkboxes[apiAggKey] &&
//                 currentSelections.checkboxes[apiAggKey].has(bucket.key)
//               ) {
//                 input.checked = true;
//               }
//               input.addEventListener('change', () => {
//                 if (!currentSelections.checkboxes[apiAggKey]) currentSelections.checkboxes[apiAggKey] = new Set();
//                 if (input.checked) {
//                   currentSelections.checkboxes[apiAggKey].add(bucket.key);
//                 } else {
//                   currentSelections.checkboxes[apiAggKey].delete(bucket.key);
//                   if (currentSelections.checkboxes[apiAggKey].size === 0) delete currentSelections.checkboxes[apiAggKey];
//                 }
//                 onAnyFilterChange();
//               });
//             });
//             card.appendChild(extraDiv);
//             const toggle = document.createElement('button');
//             toggle.className = 'text-blue-600 text-sm mt-2';
//             toggle.innerText = 'Show more';
//             toggle.onclick = () => {
//               extraDiv.classList.toggle('hidden');
//               toggle.innerText = extraDiv.classList.contains('hidden') ? 'Show more' : 'Show less';
//             };
//             card.appendChild(toggle);
//           }
//           rendered = true;
//         }
//       }
//       if (!rendered) {
//         const placeholder = document.createElement('div');
//         placeholder.className = 'card-placeholder';
//         placeholder.innerText = `No data for "${apiAggKey}" in API counts, or all counts are zero.`;
//         card.appendChild(placeholder);
//       }
//       grid.appendChild(card);
//       continue;
//     }

//     const placeholder = document.createElement('div');
//     placeholder.className = 'card-placeholder';
//     placeholder.innerText = 'Unknown card config; not rendered.';
//     card.appendChild(placeholder);
//     grid.appendChild(card);
//   }
//   mainContent.appendChild(grid);
//   restoreSelectionsToUI(mainContent);
// }


// new june-13: 11:40 PM

function renderUI(currentTab) {
  const mainContent = document.getElementById('main-content');
  mainContent.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4';

  for (const cardName of tabCards[currentTab]) {
    const cardConfig = filterData[currentTab][cardName];
    const mappingKey = mapping[cardName]; // always full, with "cases."
    const apiAggKey = stripCasesPrefix(mappingKey);
    const card = document.createElement('div');
    card.className = 'bg-white p-4 rounded shadow relative';
    const header = document.createElement('div');
    header.className = 'card-header-band';
    header.innerText = cardName;
    card.appendChild(header);

    if (!apiAggKey) {
      const dbg = document.createElement('div');
      dbg.className = "card-debug";
      dbg.innerText = `Not rendered: No mapping for card "${cardName}" in field_mappings.yaml.`;
      card.appendChild(dbg);
      grid.appendChild(card);
      continue;
    }

    // ---- RANGE/RANGE+CHECKBOX CARDS ----
    if (
      typeof cardConfig === "object" &&
      cardConfig !== null &&
      !Array.isArray(cardConfig) &&
      cardConfig.type
    ) {
      const rangeRow = document.createElement('div');
      rangeRow.className = 'flex space-x-2 items-end mb-2';
      const minGroup = document.createElement('div');
      minGroup.innerHTML = `<label class="text-xs text-gray-500">${cardConfig.min_label || "Min"}</label>
          <input type="number" placeholder="${cardConfig.min_label || "Min"}" class="min-input border rounded w-20 p-1 focus:ring focus:ring-blue-200" min="${cardConfig.min}" max="${cardConfig.max}">`;
      rangeRow.appendChild(minGroup);
      const dash = document.createElement('span');
      dash.className = 'mx-1 text-gray-400';
      dash.innerText = '—';
      rangeRow.appendChild(dash);
      const maxGroup = document.createElement('div');
      maxGroup.innerHTML = `<label class="text-xs text-gray-500">${cardConfig.max_label || "Max"} </label>
          <input type="number" placeholder="${cardConfig.max_label || "Max"}" class="max-input border rounded w-20 p-1 focus:ring focus:ring-blue-200" min="${cardConfig.min}" max="${cardConfig.max}">`;
      rangeRow.appendChild(maxGroup);
      if (cardConfig.unit) {
        const unitSpan = document.createElement('span');
        unitSpan.className = 'ml-2 text-xs text-gray-400';
        unitSpan.innerText = cardConfig.unit;
        rangeRow.appendChild(unitSpan);
      }
      card.appendChild(rangeRow);

      // Add robust range input listeners
      const minInput = rangeRow.querySelector('.min-input');
      const maxInput = rangeRow.querySelector('.max-input');
      [minInput, maxInput].forEach(input =>
        input.addEventListener('input', () => {
          const minVal = minInput.value;
          const maxVal = maxInput.value;
          if (minVal || maxVal) {
            currentSelections.ranges[mappingKey] = { min: minVal, max: maxVal };
          } else {
            delete currentSelections.ranges[mappingKey];
          }
          onAnyFilterChange();
        })
      );

      if (cardConfig.type === "range+checkboxes" && Array.isArray(cardConfig.popular)) {
        const popDiv = document.createElement('div');
        popDiv.className = 'popular-range-row flex flex-wrap gap-2 mt-2';

        let buckets = [];
        if (apiAggs[apiAggKey] && apiAggs[apiAggKey].buckets)
          buckets = apiAggs[apiAggKey].buckets;

        const cardKey = `${currentTab}-${cardName}`.replace(/[^a-zA-Z0-9_-]/g, '');

        cardConfig.popular.forEach((pop, i) => {
          let popCount = 0;
          let popPct = 0;
          if (buckets.length) {
            popCount = buckets.filter(
              b => b.key !== "_missing" && b.doc_count > 0 &&
                (Number(b.key) >= pop.min && Number(b.key) <= pop.max)
            ).reduce((acc, b) => acc + b.doc_count, 0);
            const total = buckets.filter(b => b.doc_count > 0 && b.key !== "_missing")
                                .reduce((acc, b) => acc + b.doc_count, 0);
            popPct = total ? ((popCount / total) * 100).toFixed(1) : 0;
          }
          const label = document.createElement('label');
          label.className = 'inline-flex items-center px-2 py-1 bg-gray-100 rounded cursor-pointer';
          label.innerHTML = `<input type='checkbox' data-field='${mappingKey}' class='popular-range mr-2' name='${cardKey}-pop' data-min='${pop.min}' data-max='${pop.max}' id='${cardKey}-pop-${i}'>${pop.label}
              <span class="checkbox-count">${popCount ? popCount.toLocaleString() : 0} (${popPct}%)</span>`;
          popDiv.appendChild(label);
        });
        card.appendChild(popDiv);

        // SYNC logic: listener for popular range checkboxes
        const checkboxes = popDiv.querySelectorAll('input[type=checkbox]');
        checkboxes.forEach(cb => {
          cb.addEventListener('change', () => {
            if (cb.checked) {
              checkboxes.forEach(other => { if (other !== cb) other.checked = false; });
              minInput.value = cb.dataset.min;
              maxInput.value = cb.dataset.max;
              currentSelections.ranges[mappingKey] = { min: cb.dataset.min, max: cb.dataset.max };
            } else {
              if (!minInput.value && !maxInput.value) {
                delete currentSelections.ranges[mappingKey];
              }
            }
            onAnyFilterChange();
          });
        });
      }
      grid.appendChild(card);
      continue;
    }

    // ---- STANDARD CHECKBOX CARDS ----
    if (
      Array.isArray(cardConfig) ||
      cardConfig === null ||
      (typeof cardConfig === "object" && cardConfig !== null && !cardConfig.type && Object.keys(cardConfig).length === 0)
    ) {
      let rendered = false;
      if (apiAggs[apiAggKey] && apiAggs[apiAggKey].buckets) {
        const buckets = apiAggs[apiAggKey].buckets.filter(
          b => b.doc_count > 0 && b.key !== "_missing"
        ).sort((a, b) => a.key.localeCompare(b.key, undefined, { numeric: true, sensitivity: 'base' }));
        if (buckets.length > 0) {
          const total = buckets.reduce((a, b) => a + b.doc_count, 0);
          buckets.slice(0, 6).forEach(bucket => {
            const pct = ((bucket.doc_count / total) * 100).toFixed(1);
            const row = document.createElement('div');
            row.className = "checkbox-row";
            row.innerHTML = `
              <label class="flex items-center checkbox-label">
                <input type="checkbox" data-field="${mappingKey}" data-value="${bucket.key}" class="mr-2">
                ${bucket.key}
              </label>
              <span class="checkbox-count">${bucket.doc_count.toLocaleString()} (${pct}%)</span>
            `;
            card.appendChild(row);
            const input = row.querySelector('input[type=checkbox]');
            // Restore checked state
            if (
              currentSelections.checkboxes[mappingKey] &&
              currentSelections.checkboxes[mappingKey].has(bucket.key)
            ) {
              input.checked = true;
            }
            // Robust state: update only that field/value!
            input.addEventListener('change', () => {
              if (!currentSelections.checkboxes[mappingKey]) currentSelections.checkboxes[mappingKey] = new Set();
              if (input.checked) {
                currentSelections.checkboxes[mappingKey].add(bucket.key);
              } else {
                currentSelections.checkboxes[mappingKey].delete(bucket.key);
                if (currentSelections.checkboxes[mappingKey].size === 0) delete currentSelections.checkboxes[mappingKey];
              }
              onAnyFilterChange();
            });
          });
          if (buckets.length > 6) {
            const id = `${cardName.replace(/[^a-zA-Z0-9_-]/g, '')}-more`;
            const extraDiv = document.createElement('div');
            extraDiv.id = id;
            extraDiv.className = 'hidden';
            buckets.slice(6).forEach(bucket => {
              const pct = ((bucket.doc_count / total) * 100).toFixed(1);
              const row = document.createElement('div');
              row.className = "checkbox-row";
              row.innerHTML = `
                <label class="flex items-center checkbox-label">
                  <input type="checkbox" data-field="${mappingKey}" data-value="${bucket.key}" class="mr-2">
                  ${bucket.key}
                </label>
                <span class="checkbox-count">${bucket.doc_count.toLocaleString()} (${pct}%)</span>
              `;
              extraDiv.appendChild(row);
              const input = row.querySelector('input[type=checkbox]');
              if (
                currentSelections.checkboxes[mappingKey] &&
                currentSelections.checkboxes[mappingKey].has(bucket.key)
              ) {
                input.checked = true;
              }
              input.addEventListener('change', () => {
                if (!currentSelections.checkboxes[mappingKey]) currentSelections.checkboxes[mappingKey] = new Set();
                if (input.checked) {
                  currentSelections.checkboxes[mappingKey].add(bucket.key);
                } else {
                  currentSelections.checkboxes[mappingKey].delete(bucket.key);
                  if (currentSelections.checkboxes[mappingKey].size === 0) delete currentSelections.checkboxes[mappingKey];
                }
                onAnyFilterChange();
              });
            });
            card.appendChild(extraDiv);
            const toggle = document.createElement('button');
            toggle.className = 'text-blue-600 text-sm mt-2';
            toggle.innerText = 'Show more';
            toggle.onclick = () => {
              extraDiv.classList.toggle('hidden');
              toggle.innerText = extraDiv.classList.contains('hidden') ? 'Show more' : 'Show less';
            };
            card.appendChild(toggle);
          }
          rendered = true;
        }
      }
      if (!rendered) {
        const placeholder = document.createElement('div');
        placeholder.className = 'card-placeholder';
        placeholder.innerText = `No data for "${apiAggKey}" in API counts, or all counts are zero.`;
        card.appendChild(placeholder);
      }
      grid.appendChild(card);
      continue;
    }

    const placeholder = document.createElement('div');
    placeholder.className = 'card-placeholder';
    placeholder.innerText = 'Unknown card config; not rendered.';
    card.appendChild(placeholder);
    grid.appendChild(card);
  }
  mainContent.appendChild(grid);
  restoreSelectionsToUI(mainContent);
}



async function onAnyFilterChange() {
  showLoader();
  const filters = buildGdcFilters();
  updateJsonDropdown(JSON.stringify(filters, null, 4)); // live updates to the dropdown json
  try {
    const apiJson = await fetchApiCounts(filters);
    apiAggs = apiJson.data.aggregations;
    const apiTotal = apiJson.data.pagination.total;
    updateCaseCountBar(typeof apiTotal === "number" ? apiTotal : null);
    const activeTab = tabOrder.find(tab => document.getElementById(`tab-btn-${tab}`).classList.contains('bg-blue-100'));
    renderUI(activeTab);
  } catch (e) {
    alert("Error fetching counts from GDC API!");
  } finally {
    hideLoader();
  }
//   console.log('onAnyFilterChange CALLED. currentSelections:', currentSelections);
}

function handleResetFilters() {
  showLoader();
  document.getElementById('search-bar').value = '';
  updateJsonDropdown({});
  currentSelections = { checkboxes: {}, ranges: {} };
  fetchApiCounts().then(apiJson => {
    apiAggs = apiJson.data.aggregations;
    renderUI(tabOrder[0]);
    const apiTotal = apiJson.data.pagination.total;
    updateCaseCountBar(typeof apiTotal === "number" ? apiTotal : null);
    tabOrder.forEach((tab, idx) => {
      const btn = document.getElementById(`tab-btn-${tab}`);
      btn.classList.remove('bg-blue-100', 'text-blue-600', 'text-gray-700');
      if (idx === 0) btn.classList.add('bg-blue-100', 'text-blue-600');
      else btn.classList.add('text-gray-700');
    });
    latestFilterJson = "";
  }).finally(hideLoader);
}

function handleSearch(event) {
  if (event.key === 'Enter') {
    alert(`Search triggered for: ${event.target.value}`);
  }
}


function applyFilterJsonToUI(filterJson) {
  currentSelections = { checkboxes: {}, ranges: {} };
  filterJson = JSON.parse(filterJson)
  if (!filterJson || !filterJson.content) return;
  (filterJson.content || []).forEach(clause => {
    if (clause.op === "in" && clause.content?.field && clause.content?.value) {
      const fieldPath = clause.content.field;
      if (!currentSelections.checkboxes[fieldPath]) currentSelections.checkboxes[fieldPath] = new Set();
      const vals = Array.isArray(clause.content.value) ? clause.content.value : [clause.content.value];
      vals.forEach(val => currentSelections.checkboxes[fieldPath].add(val));
    }
    if (clause.op === "range" && clause.content?.field) {
      const fieldPath = clause.content.field;
      currentSelections.ranges[fieldPath] = {
        min: clause.content.gte ?? "",
        max: clause.content.lte ?? ""
      };
    }
  });
  // Optionally show UI instantly
  const activeTab = tabOrder.find(tab => document.getElementById(`tab-btn-${tab}`).classList.contains('bg-blue-100'));
  renderUI(activeTab);
  // Always update GDC counts for *all* selections
  onAnyFilterChange();
}

// --- DOMContentLoaded --- //
window.addEventListener('DOMContentLoaded', () => {
  loadAllData();
  document.getElementById('reset-filters').addEventListener('click', handleResetFilters);
  document.getElementById('search-bar').addEventListener('keydown', handleSearch);

  document.getElementById('nl-submit').addEventListener('click', async () => {
    const query = document.getElementById('search-bar').value.trim();
    if (!query) return alert("Please enter a search query.");
    showLoader();
    try {
      const resp = await fetch(BACKEND_NL_URL, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: query})
      });
      if (!resp.ok) throw new Error("Backend error.");
      const result = await resp.json();
      // Accept result.cohort or result.filter_json, fallback to whole result
      let json = result.cohort || result.filter_json || result;
      console.log('search result output type:', typeof json)
      latestFilterJson = json; // Update global var for dropdown
      updateJsonDropdown(latestFilterJson);
      showJsonDropdown();
    } catch (e) {
      alert("Failed to parse query. " + e.message);
    } finally {
      hideLoader();
    }
  });

  document.getElementById('show-json-btn').addEventListener('click', () => {
    console.log('Show Filter JSON button clicked!');
  });

  document.getElementById('download-json').addEventListener('click', function(){
    const currentFilter = latestFilterJson ? buildGdcFilters() : null;
    let jsonStr = "";
    if (currentFilter && Object.keys(currentFilter).length > 0) {
      jsonStr = JSON.stringify(currentFilter, null, 2);
    }
    // Download as .txt
    const blob = new Blob([jsonStr], {type: "text/plain"});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cohort_filter.txt";
    document.body.appendChild(link);
    link.click();
    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    }, 100);
  });

  document.getElementById('download-tsv').addEventListener('click', async () => {
    showLoader();
    try {
      const filters = buildGdcFilters();
      const params = new URLSearchParams();
      params.set('fields', 'case_id');
      params.set('format', 'TSV');
      params.set('size', 45087 )
      if (Object.keys(filters).length > 0) params.set('filters', JSON.stringify(filters));
      const url = `${GDC_API_URL}?${params.toString()}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error("Failed to fetch TSV.");
      const blob = await resp.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = "gdc_cohort_case_ids.tsv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      alert("Download failed: " + e.message);
    } finally {
      hideLoader();
    }
  });

  
  // Button: open dropdown
  document.getElementById('show-json-btn').addEventListener('click', () => {
    updateJsonDropdown(latestFilterJson);
    showJsonDropdown();
  });

  // Button: close dropdown
  document.getElementById('close-json-dropdown').addEventListener('click', hideJsonDropdown);

  // Close on click outside dropdown
  document.addEventListener('mousedown', function(event) {
    const dropdown = document.getElementById('filter-json-dropdown');
    if (!dropdown.classList.contains('hidden') && !dropdown.contains(event.target) && event.target.id !== 'show-json-btn') {
      hideJsonDropdown();
    }
  });

  // Copy to clipboard
  document.getElementById('copy-json-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(document.getElementById('dropdown-json-content').textContent).then(() => {
      document.getElementById('copy-json-btn').textContent = "Copied!";
      setTimeout(() => document.getElementById('copy-json-btn').textContent = "Copy to Clipboard", 1000);
    });
  });

  // Populate Cohort Builder (replace this with your actual function)
  document.getElementById('populate-cohort-btn').addEventListener('click', () => {
    if (typeof applyFilterJsonToUI === "function") {
      applyFilterJsonToUI(latestFilterJson); // <-- this actually updates your filter UI!
    } else {
      alert("Filter builder integration missing!");
    }
  });

});