const stateConfig = {
  Left: { className: 'left', label: 'Left' },
  'May Leave': { className: 'may-leave', label: 'May Leave' },
  Stable: { className: 'stable', label: 'Stable' },
  'May Become Loyal': { className: 'may-become-loyal', label: 'May Become Loyal' },
  Loyal: { className: 'loyal', label: 'Loyal' }
};

const searchBtn = document.getElementById('searchBtn');
const input = document.getElementById('customerIdInput');
const resetBtn = document.getElementById('resetBtn');
const statusMessage = document.getElementById('statusMessage');
const resultArea = document.getElementById('resultArea');
const summaryGrid = document.getElementById('summaryGrid');
const customerOverview = document.getElementById('customerOverview');
const predictionPanel = document.getElementById('predictionPanel');
const statePanel = document.getElementById('statePanel');

function showStatus(message, type = 'success') {
  statusMessage.textContent = message;
  statusMessage.classList.remove('hidden', 'success', 'error');
  statusMessage.classList.add(type);
}

function hideStatus() {
  statusMessage.classList.add('hidden');
}

function renderSummary(summary) {
  const order = ['Left', 'May Leave', 'Stable', 'May Become Loyal', 'Loyal'];
  summaryGrid.innerHTML = order.map((state) => {
    const value = summary[state] ?? 0;
    const cssClass = `state-${state.toLowerCase().replace(/\s+/g, '-')}`;
    console.log(cssClass);
    return `
      <div class="summary-card ${cssClass}">
        <span class="label">${state}</span>
        <span class="count">${value}</span>
      </div>
    `;
  }).join('');
}

async function fetchSummary() {
  const response = await fetch('/api/summary');
  const summary = await response.json();
  renderSummary(summary);
}

function renderOverview(customer) {
  const fields = [
    ['Customer ID', customer.customerID],
    ['Tenure', `${customer.tenure} months`],
    ['Contract', customer.Contract],
    ['Monthly Charges', `${Number(customer.MonthlyCharges).toFixed(2)} DZD`],
    ['Total Charges', `${Number(customer.TotalCharges).toFixed(2)} DZD`],
    ['Internet Service', customer.InternetService],
    ['Payment Method', customer.PaymentMethod],
    ['Phone Service', customer.PhoneService],
    ['Online Security', customer.OnlineSecurity],
    ['Backup', customer.OnlineBackup],
    ['Device Protection', customer.DeviceProtection],
    ['Tech Support', customer.TechSupport],
    ['Streaming TV', customer.StreamingTV],
    ['Streaming Movies', customer.StreamingMovies],
    ['Paperless Billing', customer.PaperlessBilling],
    ['Partner', customer.Partner],
    ['Dependents', customer.Dependents],
    ['Senior Citizen', customer.SeniorCitizen === 1 ? 'Yes' : 'No']
  ];

  customerOverview.innerHTML = fields.map(([label, value]) => `
    <div class="info-item">
      <span class="info-label">${label}</span>
      <span class="info-value">${value ?? 'N/A'}</span>
    </div>
  `).join('');
}

function renderPrediction(prediction) {
  const label = prediction.predicted_churn === 'Yes' ? 'Churn Risk' : 'No Churn Risk';
  const pillClass = prediction.predicted_churn === 'Yes' ? 'churn' : 'no-churn';
  predictionPanel.innerHTML = `
    <div class="score-pill ${pillClass}">${label}</div>
    <div class="metric-row"><span>Churn probability</span><strong>${prediction.probability_percent}%</strong></div>
    <div class="metric-row"><span>Model prediction</span><strong>${prediction.predicted_churn}</strong></div>
  `;
}

function renderState(state, reasons) {
  const cfg = stateConfig[state] || { className: 'stable', label: state };
  statePanel.innerHTML = `
    <div class="state-banner ${cfg.className}">${cfg.label}</div>
    <ul class="reason-list">
      ${reasons.map((reason) => `<li>${reason}</li>`).join('')}
    </ul>
  `;
}

async function searchCustomer() {
  const customerId = input.value.trim();
  if (!customerId) {
    showStatus('Please enter a customer ID.', 'error');
    return;
  }

  hideStatus();
  resultArea.classList.add('hidden');
  searchBtn.disabled = true;
  searchBtn.textContent = 'Searching...';

  try {
    const response = await fetch(`/api/customer/${encodeURIComponent(customerId)}`);
    const data = await response.json();

    if (!data.found) {
      showStatus(data.message || 'Customer not found.', 'error');
      return;
    }

    renderOverview(data.customer);
    renderPrediction(data.prediction);
    renderState(data.state, data.reasons);
    resultArea.classList.remove('hidden');
    showStatus(`Customer ${customerId} loaded successfully.`, 'success');
    await fetchSummary();
  } catch (error) {
    showStatus('An unexpected error occurred while loading the customer.', 'error');
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
  }
}

function resetView() {
  input.value = '';
  hideStatus();
  resultArea.classList.add('hidden');
  customerOverview.innerHTML = '';
  predictionPanel.innerHTML = '';
  statePanel.innerHTML = '';
  fetchSummary();
}

searchBtn.addEventListener('click', searchCustomer);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    searchCustomer();
  }
});
resetBtn.addEventListener('click', resetView);

fetchSummary();
