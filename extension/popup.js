document.addEventListener('DOMContentLoaded', function() {
  const toggle = document.getElementById('toggle');
  const status = document.getElementById('status');
  const viewLogsBtn = document.getElementById('viewLogs');
  const clearLogsBtn = document.getElementById('clearLogs');
  const logsContainer = document.getElementById('logsContainer');
  const logsDiv = document.getElementById('logs');

  // Load current state
  chrome.runtime.sendMessage({ type: "GET_STATUS" }, function(response) {
    if (response && response.active !== undefined) {
      toggle.checked = response.active;
      updateStatus(response.active);
    }
  });

  // Toggle extension
  toggle.addEventListener('change', function() {
    chrome.runtime.sendMessage({ type: "TOGGLE", active: this.checked }, function(response) {
      if (response) {
        updateStatus(response.active);
      }
    });
  });

  // View logs
  viewLogsBtn.addEventListener('click', function() {
    chrome.runtime.sendMessage({ type: "GET_LOGS" }, function(response) {
      if (response && response.logs) {
        logsDiv.innerHTML = response.logs.map(log => 
          `<div class="log-entry ${log.includes('ERROR') ? 'error' : ''}">${log}</div>`
        ).join('');
        logsContainer.style.display = 'block';
      }
    });
  });

  // Clear logs
  clearLogsBtn.addEventListener('click', function() {
    chrome.runtime.sendMessage({ type: "CLEAR_LOGS" }, function(response) {
      if (response) {
        logsDiv.innerHTML = '';
        logsContainer.style.display = 'none';
      }
    });
  });

  function updateStatus(active) {
    if (active) {
      status.textContent = 'Extension Active';
      status.className = 'status active';
    } else {
      status.textContent = 'Extension Inactive';
      status.className = 'status inactive';
    }
  }
});