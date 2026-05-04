# 負責排座位與梅花座邏輯

import random

class AlgorithmMixin:
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
                    avail_seats = []
                    for s in seats:
                        parts = s['key'].split('-')
                        row, bi, p = int(parts[0]), int(parts[1]), int(parts[2])
                        
                        if bi == 0 or bi == 1:
                            if p % 2 == 0: avail_seats.append(s)
                        elif bi == 2:
                            if row % 2 == p % 2: avail_seats.append(s)
                        elif bi == 3:
                            if row == 6:
                                if p % 2 == 1: avail_seats.append(s)
                            else:
                                if p % 2 == 0: avail_seats.append(s)
                else:
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
