# 負責產生 HTML 畫面

class RendererMixin:
    def generate_html(self, selected_cls_ids, all_classrooms, seat_map, students, quincunx_on):
        students_dict = {s['id']: s for s in students}
        html_views = ''
        
        if not selected_cls_ids:
            return {
                "html_views": '<div class="no-result"><div class="icon">🏫</div>尚未排列</div>',
                "html_list": '<div class="no-result"><div class="icon">📋</div>尚未排列座位</div>'
            }
            
        for cls_id in selected_cls_ids:
            cls = next((c for c in all_classrooms if c['id'] == cls_id), None)
            if not cls: continue
            
            cmap = seat_map.get(cls_id, {})
            count = len(cmap)
            has_assigned = count > 0
            
            html_views += '<div class="cls-view"><div class="cls-center">'
            html_views += f'<div class="cls-title" style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 15px;"><span class="cls-badge">{cls["name"]}</span><span class="cls-stat">👥 {count} 人 / {self.get_capacity(cls)} 座</span></div>'
            html_views += '<div class="stage" style="width: 100%; box-sizing: border-box;">▼ 白板 / 講台 ▼</div>'
            
            t = cls.get('type')
            if t == 'fan': html_views += self.render_fan(cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned)
            elif t == 'grid': html_views += self.render_grid(cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned)
            elif t == 'el308': html_views += self.render_el308(cls, cls_id, seat_map,students_dict, quincunx_on, has_assigned)
            elif t == 'el310': html_views += self.render_el310(cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned)
            
            html_views += '</div></div>'

        rows = []
        for cls_id in selected_cls_ids:
            cmap = seat_map.get(cls_id, {})
            for k, sid in cmap.items():
                stu = students_dict.get(sid)
                rows.append(f"<tr><td>{cls_id}</td><td>{k}</td><td>{stu['id'] if stu else ''}</td><td>{stu['name'] if stu else ''}</td></tr>")
                
        if rows:
            html_list = f'<table class="exp-table"><thead><tr><th>教室</th><th>座位</th><th>學號</th><th>姓名</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        else:
            html_list = '<div class="no-result"><div class="icon">📋</div>尚未排列座位</div>'

        return {"html_views": html_views, "html_list": html_list}

    def render_seat_html(self, key, seat_num, cls_id, seat_map, students_dict, quincunx_on, has_assigned):
        sid = seat_map.get(cls_id, {}).get(key)
        stu = students_dict.get(sid)
        is_q = quincunx_on and not stu and has_assigned
        
        c = 'seat'
        if stu: c += ' occupied'
        elif is_q: c += ' quincunx'
        elif has_assigned: c += ' empty'
        
        num_html = f'<div class="seat-num">{seat_num}</div>'
        if stu:
            content = f'<div class="seat-name">{stu["name"]}</div><div class="seat-id">{stu["id"]}</div>'
        elif is_q:
            content = '<div style="font-size:.8rem;color:#ff0055">🌸</div>'
        else:
            content = '<div class="seat-id" style="color:#555">空</div>'
            
        return f'<div class="{c}" onclick="openSeatEdit(\'{key}\',\'{cls_id}\')" draggable="true" ondragstart="dragStart(\'{key}\',\'{cls_id}\')" ondragover="dragOver(event)" ondrop="drop(event,\'{key}\',\'{cls_id}\')">{num_html}{content}</div>'

    def render_fan(self, cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned):
        html = '<div class="eb106" style="display: flex; flex-direction: column; align-items: center; min-width: max-content;">'
        max_blocks = {}
        for r in cls['layout']:
            for bi, cnt in enumerate(r['blocks']):
                max_blocks[bi] = max(max_blocks.get(bi, 0), cnt)
                
        for r in cls['layout']:
            html += '<div class="eb106-row">'
            for bi, cnt in enumerate(r['blocks']):
                max_cnt = max_blocks.get(bi, 0)
                if max_cnt > 0:
                    width_style = f"width: calc({max_cnt} * (var(--seat-w) + 10px) + {max_cnt - 1} * var(--seat-gap)); flex-shrink: 0;"
                else:
                    width_style = "width: 0px;"
                
                if cnt == 0:
                    if max_cnt > 0: html += f'<div style="{width_style}"></div>'
                    continue
                
                justify = "flex-end" if bi < 2 else "flex-start"
                html += f'<div class="eb106-block" style="{width_style} justify-content: {justify};">'
                for p in range(cnt):
                    html += self.render_seat_html(f"{r['row']}-{bi}-{p}", r['row'], cls_id, seat_map, students_dict, quincunx_on, has_assigned)
                html += '</div>'
            html += '</div>'
        html += '</div>'
        return html

    def render_grid(self, cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned):
        html = '<div class="el104">'
        for ci, col in enumerate(cls['cols']):
            html += f'<div class="el104-col"><div class="el104-col-lbl">{col["label"]}</div>'
            for row in range(1, cls['rows'] + 1):
                if row < col['startRow'] or row > col['endRow']:
                    html += '<div style="height:var(--seat-h)"></div>'
                    continue
                if col['type'] == 'single':
                    html += f'<div>{self.render_seat_html(f"{row}-{ci}-0", row, cls_id, seat_map, students_dict, quincunx_on, has_assigned)}</div>'
                else:
                    html += f'<div class="desk">{self.render_seat_html(f"{row}-{ci}-0", row, cls_id, seat_map, students_dict, quincunx_on, has_assigned)}{self.render_seat_html(f"{row}-{ci}-1", row, cls_id, seat_map, students_dict, quincunx_on, has_assigned)}</div>'
            html += '</div>'
        html += '</div>'
        return html

    def render_el308(self, cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned):
        html = '<div style="display:inline-flex;flex-direction:column;align-items:stretch">'
        html += '<div style="display:flex;justify-content:flex-end;margin-bottom:5px"><div style="background:#1e2d3d;color:#00f0ff;border:1px solid #00f0ff;border-radius:6px;padding:3px 12px;font-size:.72rem;font-weight:700">🚪 門</div></div>'
        html += '<div style="display:flex;gap:var(--seat-gap);margin-bottom:5px;justify-content:center">'
        for col in cls['cols']:
            html += f'<div style="width:var(--seat-w);min-width:var(--seat-w);text-align:center;font-size:.7rem;font-weight:700;color:#8892b0">{col["label"]}</div>'
        html += '</div>'
        html += '<div style="display:flex;gap:var(--seat-gap);align-items:flex-start;justify-content:center">'
        for ci, col in enumerate(cls['cols']):
            html += '<div style="display:flex;flex-direction:column;gap:var(--seat-gap);align-items:center">'
            for r in range(cls['maxRows']):
                if r < col['rows']:
                    html += self.render_seat_html(f"{ci}-{r}", r + 1, cls_id, seat_map, students_dict, quincunx_on, has_assigned)
                else:
                    html += '<div style="width:var(--seat-w);height:var(--seat-h)"></div>'
            html += '</div>'
        html += '</div></div>'
        return html

    def render_el310(self, cls, cls_id, seat_map, students_dict, quincunx_on, has_assigned):
        e_seg_idx = next((i for i, s in enumerate(cls['segments']) if s.get('label') == 'E排'), -1)
        e_seg = cls['segments'][e_seg_idx] if e_seg_idx >= 0 else None

        hdr = '<div class="el310-headers">'
        for seg in cls['segments']:
            if seg.get('type') == 'aisle':
                hdr += '<div style="width:24px;min-width:24px"></div>'
                continue
            if seg.get('label') == 'E排': continue
            if seg.get('type') == 'teacher':
                hdr += '<div class="el310-col-lbl" style="color:#00f0ff;font-size:.65rem">教師機<br>/E排</div>'
            else:
                hdr += f'<div class="el310-col-lbl">{seg.get("label", "")}</div>'
        hdr += '</div>'

        body = '<div class="el310-body">'
        slot_h = 68 
        for si, seg in enumerate(cls['segments']):
            if seg.get('type') == 'aisle':
                total_h = cls['maxRows'] * slot_h - 6
                body += f'<div style="width:24px;min-width:24px;height:{total_h}px;background:repeating-linear-gradient(180deg,#1e2d3d 0px,#1e2d3d 3px,transparent 3px,transparent 10px);border-radius:4px;opacity:.5"></div>'
                continue
            if seg.get('label') == 'E排': continue
            
            body += '<div class="el310-col">'
            if seg.get('type') == 'teacher':
                body += '<div class="teacher-pc">💻<br>教師機</div>'
                if e_seg:
                    for r in range(e_seg['rows']):
                        body += self.render_seat_html(f"{e_seg_idx}-{r}", r + 1, cls_id, seat_map, students_dict, quincunx_on, has_assigned)
                    used = 1 + e_seg['rows']
                    if used < cls['maxRows']: body += f'<div style="height:{(cls["maxRows"] - used) * slot_h - 6}px"></div>'
            else:
                start = seg.get('startSlot', 1)
                if start > 1: body += f'<div style="height:{(start - 1) * slot_h}px"></div>'
                for r in range(seg['rows']):
                    body += self.render_seat_html(f"{si}-{r}", r + 1, cls_id, seat_map, students_dict, quincunx_on, has_assigned)
                used = (start - 1) + seg['rows']
                if used < cls['maxRows']: body += f'<div style="height:{(cls["maxRows"] - used) * slot_h - 6}px"></div>'
            body += '</div>'
        body += '</div>'
        return f'<div class="el310-outer">{hdr}{body}</div>'
