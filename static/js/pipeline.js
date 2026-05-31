// 檢測管線管理邏輯 (Pipeline & Layer Management)
// 負責管線模式下的算子堆疊操作、順序調整及多層級視圖的渲染。

// ==========================================
// 1. 圖層堆疊操作 (Layer Stack Operations)
// ==========================================

/**
 * 從下拉選單新增一個新算子圖層至堆疊末尾。
 */
function addNewLayerFromSelect() {
    const select = document.getElementById('newLayerSelect');
    const type = select.value;
    const name = select.options[select.selectedIndex].text;
    
    // 產生唯一 ID 並初始化參數
    const layerId = 'layer_' + Date.now();
    layerStack.push({
        id: layerId,
        type: type,
        name: name,
        params: {},
        enabled: true // 預設該節點為啟用狀態
    });
    
    selectLayer(layerId); // 自動選中新圖層
    renderTopLayers();    // 更新左側清單 UI
    debouncedProcessImage();
}

/**
 * 移除指定 ID 的圖層。
 */
function removeLayer(id, event) {
    event.stopPropagation();
    layerStack = layerStack.filter(layer => layer.id !== id);
    
    // 若被刪除的是當前選中層，則將選中項移至最後一層
    if (activeLayerId === id) {
        activeLayerId = layerStack.length > 0 ? layerStack[layerStack.length - 1].id : null;
    }
    
    renderTopLayers();
    if (activeLayerId) selectLayer(activeLayerId);
    else paramPanel.style.display = 'none';
    
    debouncedProcessImage();
}

/**
 * 調整圖層在堆疊中的順序 (上移/下移)。
 */
function moveLayer(index, direction, event) {
    event.stopPropagation();
    if (direction === -1 && index > 0) {
        // 交換位置
        [layerStack[index - 1], layerStack[index]] = [layerStack[index], layerStack[index - 1]];
    } else if (direction === 1 && index < layerStack.length - 1) {
        [layerStack[index + 1], layerStack[index]] = [layerStack[index], layerStack[index + 1]];
    }
    renderTopLayers();
    debouncedProcessImage();
}

/**
 * 切換圖層的啟用/停用狀態 (Bypass)
 */
function toggleLayerEnable(id, event) {
    event.stopPropagation();
    const layer = layerStack.find(l => l.id === id);
    if (layer) {
        layer.enabled = layer.enabled === false ? true : false;
        renderTopLayers();
        debouncedProcessImage(); // 重新計算管線
    }
}

/**
 * 選中特定圖層，並載入其參數至面板。
 */
function selectLayer(id) {
    if (currentMode !== 'stack') return;
    
    saveCurrentParamsToLayer(); // 儲存舊層參數
    activeLayerId = id;
    renderTopLayers();
    
    const layer = layerStack.find(l => l.id === id);
    if (layer) {
        showParamGroupForType(layer.type);
        loadParamsIntoDOM(layer);
        
        // 同步更新視圖為該節點的輸入/輸出
        const index = layerStack.findIndex(l => l.id === id);
        if (currentPipelineData && currentPipelineData[index]) {
            renderNodeView(index);
        }
    }
}

/**
 * 渲染左側管線算子清單 UI。
 */
function renderTopLayers() {
    const list = document.getElementById('topLayerList');
    if (!list) return;
    list.innerHTML = '';

    layerStack.forEach((layer, index) => {
        const isEnabled = layer.enabled !== false; // 判斷是否啟用
        const item = document.createElement('div');
        item.className = `top-layer-item ${layer.id === activeLayerId ? 'active' : ''}`;
        
        item.onclick = () => selectLayer(layer.id);
        item.style.cursor = 'pointer';
        
        // 🌟 修改：換成極簡的幾何圓點 (● / ○)，並加上顏色標示
        item.innerHTML = `
            <div class="top-layer-name" style="flex:1; ${!isEnabled ? 'text-decoration: line-through; color: #888;' : ''}">
                ${index + 1}. ${layer.name}
            </div>
            <div class="top-layer-actions">
                <button onclick="toggleLayerEnable('${layer.id}', event)" title="啟用/停用" style="background:none; border:none; font-size:1.1rem; cursor:pointer; padding: 0 4px;">
                    ${isEnabled ? '<span style="color: #64ffda;">●</span>' : '<span style="color: #888;">○</span>'}
                </button>
                <button onclick="moveLayer(${index}, -1, event)" ${index === 0 ? 'disabled style="opacity:0.3"' : ''}>▲</button>
                <button onclick="moveLayer(${index}, 1, event)" ${index === layerStack.length - 1 ? 'disabled style="opacity:0.3"' : ''}>▼</button>
                <button class="btn-del" onclick="removeLayer('${layer.id}', event)">✖</button>
            </div>
        `;
        list.appendChild(item);
    });
}

// ==========================================
// 2. 管線視圖渲染 (Pipeline View Rendering)
// ==========================================

/**
 * 全局視圖：顯示整個管線的最起始原圖與最終輸出結果。
 */
function renderGlobalView() {
    activeLayerId = null;
    renderTopLayers(); 
    paramPanel.style.display = 'none'; 
    
    document.getElementById('originalImage').src = globalInputUrl;
    document.getElementById('processedImage').src = globalOutputUrl;
    
    document.getElementById('originalHistogram').src = "data:image/png;base64," + globalOriginalHistogram;
    document.getElementById('processedHistogram').src = "data:image/png;base64," + globalProcessedHistogram;
    document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + globalOriginalSpectrum;
    document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + globalProcessedSpectrum;
    
    document.getElementById('histogramSection').style.display = 'block';
    document.getElementById('spectrumSection').style.display = 'block';
    
    // 在 Steps 區域列出所有節點的處理結果快照
    const stepsContainer = document.getElementById('stepsContainer');
    const stepsGrid = document.getElementById('stepsGrid');
    stepsGrid.innerHTML = ''; 
    
    if (currentPipelineData.length > 0) {
        stepsContainer.style.display = 'block';
        currentPipelineData.forEach((node, index) => {
            const isEnabled = layerStack[index].enabled !== false;
            const stepBox = document.createElement('div');
            stepBox.className = 'step-box';
            
            // 🌟 修改：縮圖區的標示改為簡潔的文字標籤，停用時加上灰階濾鏡
            stepBox.innerHTML = `
                <h5 style="${!isEnabled ? 'color: #888;' : ''}">
                    [ Node ${index + 1} ] ${node.operation_name} 
                    ${!isEnabled ? '<span style="color: #ff5252; font-size: 0.8em; margin-left: 4px;">(Bypassed)</span>' : ''}
                </h5>
                <img src="data:image/jpeg;base64,${node.output_img}" alt="Node Output" style="${!isEnabled ? 'opacity: 0.3; filter: grayscale(100%);' : ''}">
            `;
            stepBox.style.cursor = 'pointer';
            stepBox.onclick = () => selectLayer(layerStack[index].id);
            stepsGrid.appendChild(stepBox);
        });
    } else {
        stepsContainer.style.display = 'none';
    }
}

/**
 * 節點視圖：詳細觀察管線中特定步驟的「輸入」與「輸出」。
 */
function renderNodeView(nodeIndex) {
    const nodeData = currentPipelineData[nodeIndex];
    if (!nodeData) return;
    
    document.getElementById('originalImage').src = "data:image/jpeg;base64," + nodeData.input_img;
    document.getElementById('processedImage').src = "data:image/jpeg;base64," + nodeData.output_img;
    
    document.getElementById('originalHistogram').src = "data:image/png;base64," + nodeData.original_histogram;
    document.getElementById('processedHistogram').src = "data:image/png;base64," + nodeData.processed_histogram;
    document.getElementById('originalSpectrum').src = "data:image/jpeg;base64," + nodeData.original_spectrum;
    document.getElementById('processedSpectrum').src = "data:image/jpeg;base64," + nodeData.processed_spectrum;
    
    document.getElementById('histogramSection').style.display = 'block';
    document.getElementById('spectrumSection').style.display = 'block'; 

    const stepsContainer = document.getElementById('stepsContainer');
    const stepsGrid = document.getElementById('stepsGrid');
    stepsGrid.innerHTML = ''; 
    
    if (nodeData.steps && nodeData.steps.length > 0) {
        stepsContainer.style.display = 'block'; 
        nodeData.steps.forEach(step => {
            const stepBox = document.createElement('div');
            stepBox.className = 'step-box';
            stepBox.innerHTML = `
                <h5>[ ${step.name} ]</h5>
                <img src="data:image/jpeg;base64,${step.image}" alt="${step.name}">
            `;
            stepsGrid.appendChild(stepBox);
        });
    } else {
        stepsContainer.style.display = 'none'; 
    }
}
