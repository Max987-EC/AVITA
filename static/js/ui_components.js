// 獨立 UI 組件邏輯 (Interactive UI Components)
// 負責預設圖片載入、圖片放大鏡 (縮放/平移) 以及 WebRTC 實時相機擷取功能。

// ==========================================
// 1. 預設測試圖片載入 (Preset Images Loading)
// ==========================================

/**
 * 從伺服器靜態目錄抓取測試圖片，並模擬檔案上傳行為。
 * @param {string} filename - 圖片檔名 (位於 static/samples/)
 */
async function loadPresetImage(filename) {
    const loading = document.getElementById('loadingIndicator');
    if(loading) loading.classList.add('show');
    
    try {
        const response = await fetch(`/static/samples/${filename}`);
        if (!response.ok) throw new Error("找不到圖片檔案");
        
        const blob = await response.blob();
        // 將下載的 Blob 封裝成 File 物件
        const file = new File([blob], filename, { type: blob.type });
        
        // 透過 DataTransfer 將檔案注入到隱藏的 <input type="file">
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        const imageInput = document.getElementById('imageInput');
        imageInput.files = dataTransfer.files;
        
        // 手動觸發 change 事件以啟動影像處理流程
        imageInput.dispatchEvent(new Event('change'));
        
    } catch (error) {
        console.error("載入預設圖片失敗:", error);
        alert(`無法載入 ${filename}！請確認檔案路徑正確。`);
    } finally {
        if(loading) loading.classList.remove('show');
    }
}

// ==========================================
// 2. 圖片放大鏡功能 (Image Modal: Zoom & Pan)
// ==========================================
let currentScale = 1;
let isDragging = false;
let startX, startY, translateX = 0, translateY = 0;

/**
 * 開啟圖片放大模態框。
 * @param {string} imgSrc - 圖片來源 Base64 或 URL
 * @param {string} captionText - 圖片說明文字
 */
function openImageModal(imgSrc, captionText) {
    const imageModal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImage");
    const caption = document.getElementById("modalCaption");
    
    if (!imageModal || !modalImg) return;

    imageModal.style.display = "block";
    modalImg.src = imgSrc;
    caption.innerHTML = captionText || "影像預覽";
    
    // 重置縮放與位移狀態
    currentScale = 1;
    translateX = 0;
    translateY = 0;
    modalImg.style.transform = `translate(0px, 0px) scale(1)`;
    
    setTimeout(() => { modalImg.style.cursor = 'grab'; }, 300);
}

/**
 * 關閉放大模態框。
 */
function closeImageModal() {
    const imageModal = document.getElementById("imageModal");
    if (imageModal) imageModal.style.display = "none";
}

// 事件監聽：點擊頁面上任何影像物件時啟動放大鏡
document.addEventListener('click', function(e) {
    if (e.target.tagName === 'IMG' && (e.target.closest('.image-box') || e.target.closest('.step-box'))) {
        if (e.target.src && e.target.src.length > 10) {
            openImageModal(e.target.src, e.target.alt);
        }
    } else if (e.target.id === 'imageModal' || e.target.classList.contains('close-modal')) {
        closeImageModal();
    }
});

// 滾輪縮放邏輯
document.addEventListener('wheel', function(e) {
    if (e.target.id === 'modalImage') {
        e.preventDefault(); 
        const zoomSpeed = 0.15;
        currentScale += (e.deltaY < 0) ? zoomSpeed : -zoomSpeed;
        currentScale = Math.min(Math.max(0.5, currentScale), 15); // 限制縮放範圍 0.5x ~ 15x
        e.target.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentScale})`;
    }
}, { passive: false });

// 拖動位移邏輯 (滑鼠按下)
document.addEventListener('mousedown', function(e) {
    if (e.target.id === 'modalImage') {
        e.preventDefault();
        isDragging = true;
        startX = e.clientX - translateX;
        startY = e.clientY - translateY;
        e.target.style.cursor = 'grabbing';
    }
});

// 拖動位移邏輯 (移動中)
document.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    const modalImg = document.getElementById("modalImage");
    if (modalImg) {
        e.preventDefault();
        translateX = e.clientX - startX;
        translateY = e.clientY - startY;
        modalImg.style.transform = `translate(${translateX}px, ${translateY}px) scale(${currentScale})`;
    }
});

// 停止拖動
document.addEventListener('mouseup', function() {
    if (isDragging) {
        isDragging = false;
        const modalImg = document.getElementById("modalImage");
        if (modalImg) modalImg.style.cursor = 'grab';
    }
});

// 鍵盤快捷鍵：Esc 關閉模態框
document.addEventListener('keydown', function(e) {
    if (e.key === "Escape") closeImageModal();
});

// ==========================================
// 3. 實時相機擷取 (WebRTC Camera Capture)
// ==========================================
let currentStream = null;

/**
 * 開啟相機模態框並列出可用設備。
 */
async function openCamera() {
    document.getElementById('cameraModal').style.display = 'block';
    await getCameras();
}

/**
 * 關閉相機並釋放硬體資源。
 */
function closeCamera() {
    document.getElementById('cameraModal').style.display = 'none';
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
}

/**
 * 獲取系統所有攝影機設備並填入選單。
 */
async function getCameras() {
    try {
        // 先請求一次權限，否則無法獲取設備標籤 (Label)
        const initialStream = await navigator.mediaDevices.getUserMedia({ video: true });
        initialStream.getTracks().forEach(track => track.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        const cameraSelect = document.getElementById('cameraSelect');
        cameraSelect.innerHTML = ''; 
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `攝影機 ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        // 預設啟動第一個攝影機
        if (videoDevices.length > 0) startStream(videoDevices[0].deviceId);
        cameraSelect.onchange = () => startStream(cameraSelect.value);

    } catch (err) {
        console.error("無法存取攝影機:", err);
        alert("無法存取攝影機，請確認瀏覽器權限已開啟！");
    }
}

/**
 * 啟動特定設備的影像串流。
 */
async function startStream(deviceId) {
    if (currentStream) currentStream.getTracks().forEach(track => track.stop());
    
    const constraints = {
        video: { 
            deviceId: deviceId ? { exact: deviceId } : undefined, 
            width: { ideal: 1920 }, 
            height: { ideal: 1080 } 
        }
    };

    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('cameraVideo');
        video.srcObject = currentStream;
    } catch (err) {
        console.error("啟動影像流失敗:", err);
    }
}

/**
 * 擷取目前畫面幀並轉換為圖片檔案。
 */
document.addEventListener('click', function(e) {
    const captureBtn = e.target.closest('#captureBtn');
    if (!captureBtn) return;

    const video = document.getElementById('cameraVideo');
    const canvas = document.getElementById('cameraCanvas');
    if (!video || !canvas) return;

    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // 水平鏡像處理 (符合自拍預覽習慣)
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 轉換為 Blob 並自動填充至上傳欄位
    canvas.toBlob((blob) => {
        const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        const fileInput = document.querySelector('input[type="file"]');
        
        if (fileInput) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
        closeCamera();
    }, 'image/jpeg', 0.95);
});
