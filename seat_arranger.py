import random
import csv
import io

class SeatArranger:
    """
    學生座位排列核心邏輯模組 (Backend-Driven UI)
    """
    CLASSROOMS = [
        { "id": 'EB106', "name": 'EB106', "type": 'fan', "layout": [ {"row":1, "blocks":[2,2,2,0]}, {"row":2, "blocks":[3,3,3,3]}, {"row":3, "blocks":[4,4,4,3]}, {"row":4, "blocks":[5,5,5,3]}, {"row":5, "blocks":[6,6,6,3]}, {"row":6, "blocks":[7,7,7,2]} ] },
        { "id": 'EL102', "name": 'EL102', "type": 'grid', "rows": 8, "cols": [ {"label":'A排', "type":'desk', "startRow":1, "endRow":5}, {"label":'B排', "type":'desk', "startRow":1, "endRow":5}, {"label":'C排', "type":'desk', "startRow":1, "endRow":5}, {"label":'D排', "type":'desk', "startRow":1, "endRow":4} ] },
        { "id": 'EL103', "name": 'EL103', "type": 'grid', "rows": 9, "cols": [ {"label":'A排', "type":'desk', "startRow":1, "endRow":9}, {"label":'B排', "type":'desk', "startRow":1, "endRow":9}, {"label":'C排', "type":'desk', "startRow":1, "endRow":9}, {"label":'D排', "type":'desk', "startRow":1, "endRow":9}, {"label":'側走道', "type":'single', "startRow":1, "endRow":7} ] },
        { "id": 'EL104', "name": 'EL104', "type": 'grid', "rows": 9, "cols": [ {"label":'側走道', "type":'single', "startRow":2, "endRow":7}, {"label":'A排', "type":'desk', "startRow":1, "endRow":9}, {"label":'B排', "type":'desk', "startRow":1, "endRow":9}, {"label":'C排', "type":'desk', "startRow":1, "endRow":9}, {"label":'D排', "type":'desk', "startRow":1, "endRow":9} ] },
        { "id": 'EL105', "name": 'EL105', "type": 'grid', "rows": 8, "cols": [ {"label":'A排', "type":'single', "startRow":1, "endRow":5}, {"label":'B排', "type":'single', "startRow":2, "endRow":6}, {"label":'C排', "type":'single', "startRow":2, "endRow":6}, {"label":'D排', "type":'single', "startRow":2, "endRow":6}, {"label":'E排', "type":'single', "startRow":2, "endRow":6}, {"label":'F排', "type":'single', "startRow":2, "endRow":6}, {"label":'G排', "type":'single', "startRow":2, "endRow":6}, {"label":'H排', "type":'single', "startRow":2, "endRow":6} ] },
        { "id": 'EL106', "name": 'EL106', "type": 'grid', "rows": 9, "cols": [ {"label":'側走道', "type":'single', "startRow":1, "endRow":7}, {"label":'A排', "type":'desk', "startRow":1, "endRow":9}, {"label":'B排', "type":'desk', "startRow":1, "endRow":9}, {"label":'C排', "type":'desk', "startRow":1, "endRow":9}, {"label":'D排', "type":'desk', "startRow":1, "endRow":9} ] },
        { "id": 'EL308', "name": 'EL308', "type": 'el308', "maxRows": 6, "cols": [ { "label":'A排', "rows":5 }, { "label":'B排', "rows":6 }, { "label":'C排', "rows":6 }, { "label":'D排', "rows":6 }, { "label":'E排', "rows":6 }, { "label":'F排', "rows":3 } ] },
        { "id": 'EL310', "name": 'EL310', "type": 'el310', "maxRows": 12, "segments": [ { "label":'A排', "type":'single', "startSlot":1, "rows":12 }, { "label":'', "type":'aisle' }, { "label":'B排', "type":'single', "startSlot":2, "rows":10 }, { "label":'C排', "type":'single', "startSlot":2, "rows":10 }, { "label":'D排', "type":'single', "startSlot":2, "rows":10 }, { "label":'', "type":'aisle' }, { "label":'教師機', "type":'teacher', "startSlot":1, "rows":1 }, { "label":'E排', "type":'single', "startSlot":2, "rows":9 }, { "label":'F排', "type":'single', "startSlot":2, "rows":8 } ] }
    ]

    def handle_request(self, data):
        """統一處理前端的各種請求"""
        action = data.get('action', 'arrange')
        custom_cls = data.get('customClassrooms', [])
        all_classrooms = self.CLASSROOMS + custom_cls
        
        if action == 'get_config':
            for c in all_classrooms:
                c['capacity'] = self.get_capacity(c)
            return {"classrooms": all_classrooms}
            
        elif action == 'arrange':
            return self.process_arrangement(data, all_classrooms)
            
        elif action == 'render_only':
            return self.generate_html(
                data.get('selectedCls', []), 
                all_classrooms, 
                data.get('seatMap', {}), 
                data.get('students', []), 
                data.get('quincunx', False)
            )

    # ==========================================
    # ⚙️ 核心排列與 HTML 渲染邏輯
    # ==========================================
    def get_capacity(self, cls):
        t = cls.get('type')
        if t == 'fan': return sum(sum(r['blocks']) for r in cls['layout'])
        if t == 'grid':
            c = 0
            for col in cls['cols']:
                rows = col['endRow'] - col['startRow'] + 1
                c += rows if col['type'] == 'single' else rows * 2
            return c
        if t == 'el308': return sum(c['rows'] for c in cls['cols'])
        if t == 'el310':
            return sum(seg['rows'] for seg in cls['segments'] if seg['type'] not in ('aisle', 'teacher'))
        return 0

    def build_seat_grid(self, cls):
        seats = []
        t = cls.get('type')
        if t == 'fan':
            for r in cls['layout']:
                col_idx = 0
                for bi, cnt in enumerate(r['blocks']):
                    if cnt == 0:
                        col_idx += 8
                        continue
                    for p in range(cnt):
                        seats.append({"key": f"{r['row']}-{bi}-{p}", "gridRow": r['row'], "gridCol": col_idx + p})
                    col_idx += cnt + 2
        elif t == 'grid':
            col_idx = 0
            for ci, col in enumerate(cls['cols']):
                for row in range(col['startRow'], col['endRow'] + 1):
                    if col['type'] == 'single':
                        seats.append({"key": f"{row}-{ci}-0", "gridRow": row, "gridCol": col_idx})
                    else:
                        seats.append({"key": f"{row}-{ci}-0", "gridRow": row, "gridCol": col_idx})
                        seats.append({"key": f"{row}-{ci}-1", "gridRow": row, "gridCol": col_idx + 1})
                col_idx += 2 if col['type'] == 'single' else 3
        elif t == 'el308':
            for ci, col in enumerate(cls['cols']):
                for r in range(col['rows']):
                    seats.append({"key": f"{ci}-{r}", "gridRow": r + 1, "gridCol": ci})
        elif t == 'el310':
            col_idx = 0
            for si, seg in enumerate(cls['segments']):
                if seg['type'] in ('aisle', 'teacher'):
                    col_idx += 2
                    continue
                start = seg.get('startSlot', 1)
                for r in range(seg['rows']):
                    seats.append({"key": f"{si}-{r}", "gridRow": start + r, "gridCol": col_idx})
                col_idx += 2
        return seats

    def process_arrangement(self, data, all_classrooms):
        students_raw = data.get('students_raw', '')
        mode = data.get('mode', 'order')
        quincunx = data.get('quincunx', False)
        selected_cls_ids = data.get('selectedCls', [])

        lines = [line.strip() for line in students_raw.split('\n') if line.strip()]
        students = []
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 2:
                students.append({"id": parts[0], "name": " ".join(parts[1:])})
            else:
                students.append({"id": f"S{str(i+1).zfill(3)}", "name": parts[0]})

        if mode == 'random': random.shuffle(students)
        elif mode == 'name': students.sort(key=lambda x: x['name'])
        else: students.sort(key=lambda x: x['id'])

        seat_map = {}
        stu_idx = 0

        for cls_id in selected_cls_ids:
            cls = next((c for c in all_classrooms if c['id'] == cls_id), None)
            if not cls: continue
            
            seats = self.build_seat_grid(cls)
            if quincunx:
                if cls.get('type') == 'fan':
                    # 🌟 EB106 專屬：完全依照客製化規則分配 (修正 0 與 1 的起算差異)
                    avail_seats = []
                    for s in seats:
                        parts = s['key'].split('-')
                        row = int(parts[0])  # 排數 (1, 2, 3...)
                        bi = int(parts[1])   # 區塊索引 (0, 1, 2, 3)
                        p = int(parts[2])    # 座位索引 (0, 1, 2...)
                        
                        if bi == 0 or bi == 1:
                            # 規則 1：區塊 0, 1 坐第 1, 3, 5 個位子 (程式索引 p=0, 2, 4)
                            if p % 2 == 0:
                                avail_seats.append(s)
                                
                        elif bi == 2:
                            # 規則 2：區塊 3 奇數排坐偶數位 (p=1,3,5)，偶數排坐奇數位 (p=0,2,4)
                            if row % 2 == p % 2:
                                avail_seats.append(s)
                                
                        elif bi == 3:
                            # 規則 3：區塊 4 坐第 1, 3, 5 個位子 (p=0,2,4)，只有第六排坐偶數位 (p=1,3,5)
                            if row == 6:
                                if p % 2 == 1:
                                    avail_seats.append(s)
                            else:
                                if p % 2 == 0:
                                    avail_seats.append(s)
                else:
                    # 其他方形教室維持原本的「上下左右防碰撞」邏輯
                    occupied = set()
                    avail_seats = []
                    for s in seats:
                        r, c = s['gridRow'], s['gridCol']
                        neighbors = [f"{r-1},{c}", f"{r+1},{c}", f"{r},{c-1}", f"{r},{c+1}"]
                        if not any(n in occupied for n in neighbors):
                            avail_seats.append(s)
                            occupied.add(f"{r},{c}")
            else:
                avail_seats = seats

            seat_map[cls_id] = {}
            for s in avail_seats:
                if stu_idx >= len(students): break
                seat_map[cls_id][s['key']] = students[stu_idx]['id']
                stu_idx += 1

        html_result = self.generate_html(selected_cls_ids, all_classrooms, seat_map, students, quincunx)
        
        return {
            "seatMap": seat_map,
            "students": students,
            "assignedCount": stu_idx,
            "html_views": html_result["html_views"],
            "html_list": html_result["html_list"]
        }

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
            
            html_views += '<div class="cls-view">'
            
            # 🌟 1. 讓 cls-center 包住所有東西（包含標題）
            html_views += '<div class="cls-center">'
            
            # 🌟 2. 將標題移到這裡，並加上 width: 100% 與 justify-content: center 讓它置中
            html_views += f'<div class="cls-title" style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 15px;"><span class="cls-badge">{cls["name"]}</span><span class="cls-stat">👥 {count} 人 / {self.get_capacity(cls)} 座</span></div>'
            
            # 白板保持不變
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
                
                # 🌟 修正 3：寬度加上 10px (包含邊框與外距 Margin)，確保座位絕對不會滿出來
                if max_cnt > 0:
                    width_style = f"width: calc({max_cnt} * (var(--seat-w) + 10px) + {max_cnt - 1} * var(--seat-gap)); flex-shrink: 0;"
                else:
                    width_style = "width: 0px;"
                
                if cnt == 0:
                    if max_cnt > 0:
                        html += f'<div style="{width_style}"></div>'
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
        slot_h = 68 # var(--seat-h) + var(--seat-gap)
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
