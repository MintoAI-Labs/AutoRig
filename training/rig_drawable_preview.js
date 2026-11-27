'use strict';
// Extend the generated audit copy. The original comparison file is untouched.
const isolation = document.createElement('div');
isolation.className = 'bar';
isolation.innerHTML = '<label>Nhóm drawable<select id="drawable-group"></select></label>' +
  '<label>Drawable<select id="drawable-id"></select></label>' +
  '<p id="drawable-details" role="status"></p>';
document.querySelector('.bar').after(isolation);
const groupSelect = document.querySelector('#drawable-group');
const drawableSelect = document.querySelector('#drawable-id');
const drawableDetails = document.querySelector('#drawable-details');

function option(select, value, label) {
  const item = document.createElement('option');
  item.value = value;
  item.textContent = label;
  select.append(item);
}
function fillDrawables(drawables) {
  drawableSelect.replaceChildren();
  option(drawableSelect, '', 'Cả nhóm');
  drawables.forEach((item, index) => {
    if (!groupSelect.value || groupSelect.value === item.category) {
      option(drawableSelect, index, item.id + (item.part ? ' · ' + item.part : ''));
    }
  });
}
let previousModel;
const drawComparison = draw;
draw = function () {
  if (!current) return;
  const drawables = current.model.drawables;
  if (current.model !== previousModel) {
    previousModel = current.model;
    groupSelect.replaceChildren();
    option(groupSelect, '', 'Tất cả');
    for (const label of ['Mắt','Mặt','Tóc','Thân','Khác']) {
      if (drawables.some(item => item.category === label)) option(groupSelect, label, label);
    }
    fillDrawables(drawables);
  }
  drawComparison();
  const selection = drawableSelect.value === '' ? null : drawables[Number(drawableSelect.value)];
  if (selection) {
    const {points} = current;
    const report = points.slice(1).map(prediction => {
      let max = 0;
      for (let i = selection.start * 2; i < selection.end * 2; i++) {
        max = Math.max(max, Math.abs(prediction[i] - points[0][i]));
      }
      return (100 * max).toFixed(5) + '%';
    });
    drawableDetails.textContent = selection.id + ' · ' + selection.category +
      (selection.label ? ' · ' + selection.label : '') + ' · đỉnh ' +
      selection.start + '–' + (selection.end - 1) + ' · max theo kích thước model: ' + report.join(' / ');
  } else {
    drawableDetails.textContent = 'Lọc hình để xem lớp. Bảng điểm bên dưới vẫn chấm toàn mô hình.';
  }
};

for (const renderer of renderers) {
  const originalDraw = renderer.draw.bind(renderer);
  renderer.draw = (points, order, opacity, wire) => {
    const visible = order.filter(index => {
      if (drawableSelect.value !== '') return index === Number(drawableSelect.value);
      return !groupSelect.value || current.model.drawables[index].category === groupSelect.value;
    });
    originalDraw(points, visible, opacity, wire);
  };
}
groupSelect.onchange = () => {if (current) {fillDrawables(current.model.drawables);draw();}};
drawableSelect.onchange = () => draw();
mode.value = 'wire';
draw();
