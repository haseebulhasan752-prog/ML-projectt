/**
 * predict.js — Dynamic form: hide the selected target field
 * Student Mental Health AI System
 */

(function () {
  'use strict';

  // Map radio value → field data-field name
  const TARGET_FIELD_MAP = {
    'CGPA':             'CGPA',
    'Stress_Level':     'Stress_Level',
    'Depression_Score': 'Depression_Score',
    'Anxiety_Score':    'Anxiety_Score',
  };

  function hideTargetField(target) {
    // Show all fields first
    document.querySelectorAll('.field-wrapper').forEach(el => {
      el.classList.remove('hidden-field');
    });

    // Hide the field corresponding to selected target
    const fieldName = TARGET_FIELD_MAP[target];
    if (fieldName) {
      const wrapper = document.querySelector(`.field-wrapper[data-field="${fieldName}"]`);
      if (wrapper) {
        wrapper.classList.add('hidden-field');
        // Disable the input so it isn't submitted
        wrapper.querySelectorAll('input, select').forEach(inp => {
          inp.disabled = true;
        });
      }
    }

    // Re-enable all other field inputs
    document.querySelectorAll('.field-wrapper:not(.hidden-field) input, .field-wrapper:not(.hidden-field) select')
      .forEach(inp => { inp.disabled = false; });
  }

  function init() {
    const radios = document.querySelectorAll('input[name="target"]');
    if (!radios.length) return;

    // Initial state
    const checked = document.querySelector('input[name="target"]:checked');
    if (checked) hideTargetField(checked.value);

    // On change
    radios.forEach(radio => {
      radio.addEventListener('change', () => {
        if (radio.checked) hideTargetField(radio.value);
      });
    });

    // Form validation feedback
    const form = document.getElementById('predForm');
    if (form) {
      form.addEventListener('submit', function (e) {
        const btn = form.querySelector('button[type="submit"]');
        if (btn) {
          btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Running AI...';
          btn.disabled = true;
        }
      });
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
