function getVideoTitle() {
  const selectors = [
    'h1.title yt-formatted-string',
    'h1.ytd-video-primary-info-renderer',
    '#title h1',
    '.title.style-scope.ytd-video-primary-info-renderer'
  ];
  
  for (const selector of selectors) {
    const el = document.querySelector(selector);
    if (el && el.innerText && el.innerText.trim()) {
      return el.innerText.trim();
    }
  }
  return null;
}

function blockYouTubePlayer() {
  const playerSelectors = [
    '#movie_player',
    '.html5-video-player',
    'video',
    'ytd-player'
  ];
  
  for (const selector of playerSelectors) {
    const player = document.querySelector(selector);
    if (player) {
      const video = player.querySelector('video');
      if (video) {
        video.pause();
      }
      
      let blocker = document.getElementById("smartstudy-player-blocker");
      if (!blocker) {
        blocker = document.createElement("div");
        blocker.id = "smartstudy-player-blocker";
        blocker.style.position = "absolute";
        blocker.style.top = "0";
        blocker.style.left = "0";
        blocker.style.width = "100%";
        blocker.style.height = "100%";
        blocker.style.background = "rgba(0, 0, 0, 0.8)";
        blocker.style.color = "white";
        blocker.style.display = "flex";
        blocker.style.justifyContent = "center";
        blocker.style.alignItems = "center";
        blocker.style.fontSize = "24px";
        blocker.style.fontWeight = "bold";
        blocker.style.zIndex = "10000";
        blocker.style.textAlign = "center";
        blocker.innerHTML = "⛔ Blocked by SmartStudyTube<br><small>  This content is not study-related</small>";
        
        player.style.position = "relative";
        player.appendChild(blocker);
      }
      break;
    }
  }
}

function unblockYouTubePlayer() {
  const blocker = document.getElementById("smartstudy-player-blocker");
  if (blocker && blocker.parentNode) {
    blocker.parentNode.removeChild(blocker);
  }
}

function showOverlay(result) {
  let overlay = document.getElementById("smartstudy-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "smartstudy-overlay";
    overlay.style.position = "fixed";
    overlay.style.top = "20px";
    overlay.style.right = "20px";
    overlay.style.padding = "10px 20px";
    overlay.style.borderRadius = "10px";
    overlay.style.color = "white";
    overlay.style.fontSize = "16px";
    overlay.style.zIndex = "9999";
    overlay.style.boxShadow = "0 2px 10px rgba(0,0,0,0.3)";
    overlay.style.fontWeight = "bold";
    overlay.style.maxWidth = "300px";
    overlay.style.textAlign = "center";
    document.body.appendChild(overlay);
  }

  if (result.error) {
    overlay.style.background = "orange";
    overlay.innerText = `Error: ${result.error}`;
  } else if (result.label === "Study Content") {
    overlay.style.background = "green";
    overlay.innerText = `${result.label} (${(result.probability * 100).toFixed(1)}%)`;
    unblockYouTubePlayer();
  } else {
    overlay.style.background = "red";
    overlay.innerText = `${result.label} (${(result.probability * 100).toFixed(1)}%)`;
    if (result.probability === 0 || result.label !== "Study Content") {
      blockYouTubePlayer();
    }
  }

  setTimeout(() => {
    if (overlay && overlay.parentNode) {
      overlay.parentNode.removeChild(overlay);
    }
  }, 5000);
}

let lastTitle = "";

function checkForTitleChange() {
  const title = getVideoTitle();
  if (title && title !== lastTitle) {
    lastTitle = title;
    console.log("New title detected:", title);
    chrome.runtime.sendMessage({ type: "PREDICT_TITLE", title: title });
    
    unblockYouTubePlayer();
  }
}

setInterval(checkForTitleChange, 2000);

let observer = new MutationObserver(() => {
  checkForTitleChange();
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "PREDICTION_RESULT") {
    console.log("Received prediction:", msg.result);
    showOverlay(msg.result);
    
    if (msg.result.probability === 0 || msg.result.label !== "Study Content") {
      blockYouTubePlayer();
    }
  }
});

checkForTitleChange();