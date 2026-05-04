# 負責教室資料定義與基礎工具

class BaseArranger:
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
