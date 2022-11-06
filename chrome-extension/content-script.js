const ENDPOINT_URL = 'http://127.0.0.1:8080/token/valuation';

const tooltip = document.createElement('div');
tooltip.classList.add('smarty-nft-tooltip');
document.body.appendChild(tooltip);

const img = document.querySelector('.Image--image');
const token = document.querySelector('.item--main .item--title');
const collection = document.querySelector('.CollectionLink--link span');

fetchPrediction(img.src, token.innerText, collection.innerText).then((prediction) => {
    const predictionEl = document.createElement('span');
    predictionEl.innerText = prediction;
    tooltip.appendChild(predictionEl);

    document.body.addEventListener('mousemove', async (ev) => {
        if (ev.target === img || ev.target === token) {
            tooltip.style.top = ev.y - 20  + 'px';
            tooltip.style.left = ev.x + 20 + 'px';
            tooltip.style.opacity = '1';
        }
    },true);

    document.body.addEventListener('mouseout', (ev) => {
        if (ev.target === img || ev.target === token) {
            tooltip.style.opacity = '0';
        }
    },true);
});

// stub
async function fetchPrediction(img, token, collection) {
    return ['Bad', 'Good', 'Super'][Math.floor(Math.random() * 10 % 3)];
}
