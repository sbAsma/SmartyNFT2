(async function() {
    const tooltip = document.createElement('div');
    tooltip.classList.add('smarty-nft-tooltip');
    document.body.appendChild(tooltip);

    const {img, token, collection} = await getFeaturesContainers();

    const prediction = await fetchPrediction(img.src, token.innerText, collection.innerText)

    const predictionEl = document.createElement('span');
    predictionEl.innerText = prediction;
    tooltip.appendChild(predictionEl);

    document.body.addEventListener('mousemove', async (ev) => {
        if (ev.target === img || ev.target === token) {
            tooltip.style.top = ev.y - 20 + 'px';
            tooltip.style.left = ev.x + 20 + 'px';
            tooltip.style.opacity = '1';
        }
    }, true);

    document.body.addEventListener('mouseout', (ev) => {
        if (ev.target === img || ev.target === token) {
            tooltip.style.opacity = '0';
        }
    }, true);

})();

// stub
async function fetchPrediction(img, token, collection) {
    return ['Bad', 'Good', 'Super'][Math.floor(Math.random() * 10 % 3)];
}

function getFeaturesContainers() {
    return new Promise((resolve, reject) => {
        const timer = setInterval(function () {
            const els = {
                img: document.querySelector('.Image--image'),
                token: document.querySelector('.item--main .item--title'),
                collection: document.querySelector('.CollectionLink--link span'),
            };

            if (!Object.values(els).includes(null)) {
                clearInterval(timer);
                resolve(els);
            }

            if (document.querySelector('.AssetMedia--video')) {
                clearInterval(timer);
                reject("Asset is a video; only images are supported");
            }
        }, 500);
    })
}
