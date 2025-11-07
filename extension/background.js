let isActive = true; // Enable by default

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ logs: [], isActive: true });
  console.log("SmartStudyTube extension installed");
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "TOGGLE") {
    isActive = msg.active;
    console.log(`Extension ${isActive ? "activated" : "deactivated"}`);
    sendResponse({ status: "ok", active: isActive });
  }

  if (msg.type === "GET_STATUS") {
    sendResponse({ active: isActive });
  }

  if (msg.type === "PREDICT_TITLE" && isActive) {
    console.log("Predicting title:", msg.title);
    
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: msg.title })
    })
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        const logMsg = `[${new Date().toLocaleTimeString()}] ${msg.title} -> ${data.label} (${(data.probability * 100).toFixed(1)}%)`;
        
        chrome.storage.local.get(["logs"], result => {
          const newLogs = result.logs || [];
          newLogs.push(logMsg);
          if (newLogs.length > 100) {
            newLogs.shift();
          }
          chrome.storage.local.set({ logs: newLogs });
        });

        if (sender.tab && sender.tab.id) {
          chrome.tabs.sendMessage(sender.tab.id, { 
            type: "PREDICTION_RESULT", 
            result: data 
          });
        }
        
        console.log("Prediction result:", data);
      })
      .catch(err => {
        console.error("Prediction error:", err);
        
        const errorLog = `[${new Date().toLocaleTimeString()}] ERROR: ${err.message}`;
        chrome.storage.local.get(["logs"], result => {
          const newLogs = result.logs || [];
          newLogs.push(errorLog);
          if (newLogs.length > 100) {
            newLogs.shift();
          }
          chrome.storage.local.set({ logs: newLogs });
        });

        // Send error back to content script
        if (sender.tab && sender.tab.id) {
          chrome.tabs.sendMessage(sender.tab.id, { 
            type: "PREDICTION_RESULT", 
            result: { 
              label: "Error", 
              probability: 0.5,
              error: err.message 
            } 
          });
        }
      });
  }
  
  if (msg.type === "GET_LOGS") {
    chrome.storage.local.get(["logs"], result => {
      sendResponse({ logs: result.logs || [] });
    });
    return true;
  }
  
  if (msg.type === "CLEAR_LOGS") {
    chrome.storage.local.set({ logs: [] });
    sendResponse({ status: "logs cleared" });
  }
});