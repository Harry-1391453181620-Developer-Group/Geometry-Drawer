""" This is the main Python code of Geometry Drawer v0.0.3, welcome to check and fing errors.
    Geometry Drawer is a simple and powerful tool to draw and visualize mathematical functions and lines interactively.
    You are able to find some rules of input and dependencies in INSTRUCTIONS.md. We suggest you to read it before using this tool.
    Geometry Drawer v0.0.3 in the latest version and it updated the precision of the functions.
    The Github repository is https://github.com/Harry-1391453181620/Geometry-Drawer-v0.0.2.
                                                                                     -- Harry-1391453181620 Developer Group"""


import matplotlib.pyplot as plt
import numpy as np
import math
import re
from sympy import lambdify, symbols, Eq, solve, N
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from decimal import Decimal, getcontext

# high precision calculation
getcontext().prec = 20

# special trigonometric functions
special_trig_map = {
    'versin': '1 - cos(x)',
    'coversin': '1 - sin(x)',
    'hacoversin': '(1 - sin(x))/2',
    'exsec': 'sec(x) - 1',
    'excsc': 'csc(x) - 1'
}

INTERSECTION_TOLERANCE = Decimal('1e-4')
X_LIMIT = 100
Y_LIMIT = 10

def create_interactive_plot():
    """main function"""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # axis
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_position(('data', 0))
    plt.title('Geometry Drawer', y=1.02)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # core initing
    x = symbols('x')
    plots = {}  # stores the functions' graphs
    lines = []  # stores the line segments' graphs
    inf_lines = []  # stores the infinite lines' graphs
    def add_infinite_line(line_str):
        """Adding infinite lines, based on the exsisting axis"""
        pattern = r'draw line (\w+) \(([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\), (\w+) \(([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\) as line (\w+)(\w+)$'
        match = re.match(pattern, line_str)
        if not match:
            print("Invalid line format. Use: 'draw line A (x1, y1), B (x2, y2) as line AB'")
            return
        try:
            name1, x1_str, y1_str, name2, x2_str, y2_str, line_name1, line_name2 = match.groups()
            x1, y1 = float(x1_str), float(y1_str)
            x2, y2 = float(x2_str), float(y2_str)
            # checking the area if the axis
            x_min = float(global_limits['x_min'])
            x_max = float(global_limits['x_max'])
            y_min = float(global_limits['y_min'])
            y_max = float(global_limits['y_max'])
            # getting a larger span
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_span = (x_max - x_min) * 1.5 if (x_max - x_min) > 0 else 10
            y_span = (y_max - y_min) * 1.5 if (y_max - y_min) > 0 else 10
            t = np.linspace(-1, 1, 1000)
            scale = max(x_span, y_span)
            t = t * scale
            x_np = x1 + t * (x2 - x1)
            y_np = y1 + t * (y2 - y1)
            color = colors[(len(plots) + len(lines) + len(inf_lines)) % len(colors)]
            sizes = np.full(len(x_np), 3)
            scatter = plt.scatter(
                x_np, y_np, s=sizes, color=color,
                edgecolor=color, alpha=1, label=f"line {line_name1}{line_name2}"
            )
            inf_lines.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'x': x_np, 'y': y_np, 'color': color,
                'sizes': sizes, 'name1': name1, 'name2': name2,
                'scatter': scatter,
                'line_name': f'{line_name1}{line_name2}'
            })
            plt.legend()
            calculate_intersections()
            plt.draw()
        except Exception as e:
            print(f"Error adding infinite line: {str(e)}")
            return
    intersection_points = {}  # stores the intersection points
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c567b', '#e377c2',
        '#17becf', '#7f7f7f', '#bcbd22', "#2edda0", "#7a1eeb", "#5b0f08", "#3215eb",
        "#5C8210", "#3cb371", "#ff1493", "#12b6cb", "#053738", "#63502C", "#0E0354",
        "#3E4F50", "#A3681A", "#243D65", "#6A6A6A", "#234567", "#000000", "#FCB3B3",
        "#9F9F97", "#35A890", "#C45D23", "#21AEFD", "#3F2B4D", "#6789AB", "#61A1A1"
    ]# stores the colors of the objects
    hover_points = []  # hover points
    has_imaginary = set()  # stores functions that have non real values

    # tracking the axis
    global_limits = {
        'x_min': Decimal(str(-X_LIMIT)),
        'x_max': Decimal(str(X_LIMIT)),
        'y_min': Decimal(str(-Y_LIMIT)),
        'y_max': Decimal(str(Y_LIMIT))
    }

    def update_global_limits(new_x, new_y):
        """update the axis"""
        x_vals = [Decimal(str(val)) for val in new_x]
        y_vals = [Decimal(str(val)) for val in new_y]
        
        global_limits['x_min'] = min(global_limits['x_min'], min(x_vals))
        global_limits['x_max'] = max(global_limits['x_max'], max(x_vals))
        global_limits['y_min'] = min(global_limits['y_min'], min(y_vals))
        global_limits['y_max'] = max(global_limits['y_max'], max(y_vals))
        
        pad_x = (global_limits['x_max'] - global_limits['x_min']) * Decimal('0.1')
        pad_y = (global_limits['y_max'] - global_limits['y_min']) * Decimal('0.1')
        
        ax.set_xlim(
            float(global_limits['x_min'] - pad_x),
            float(global_limits['x_max'] + pad_x)
        )
        ax.set_ylim(
            float(global_limits['y_min'] - pad_y),
            float(global_limits['y_max'] + pad_y)
        )
        ax.set_aspect('auto')

    def parse_expression(expr_str):
        """expression intepretion"""
        # replace special trigonometric functions
        for trig_name, trig_expr in special_trig_map.items():
            expr_str = expr_str.replace(trig_name, f'({trig_expr})')
        
        # relpace log and pi
        expr_str = expr_str.replace('ln', 'log')  # supports log functions
        expr_str = expr_str.replace('log10', 'log(x, 10)')
        expr_str = expr_str.replace('√', 'sqrt')
        expr_str = expr_str.replace('π', 'np.pi')
        
        # using sympy the intepret the expressions
        transformations = (standard_transformations + (implicit_multiplication_application,))
        try:
            expr = parse_expr(expr_str, transformations=transformations)
            return lambdify('x', expr, modules=[np, {'math': math}])
        except Exception:
            return None

    def calculate_intersections():
        """calculate all the intersections between objects"""
        nonlocal intersection_points
        func_expressions = [data['expression'] for data in plots.values()]
        all_lines = []
        for data in lines:
            x1, y1 = data['x1'], data['y1']
            x2, y2 = data['x2'], data['y2']
            if x1 == x2:
                line_expr = f'x == {x1}'
            elif y1 == y2:
                line_expr = f'y == {y1}'
            else:
                k = (y2 - y1) / (x2 - x1)
                line_expr = f'({k})*(x - {x1}) + {y1}'
            all_lines.append({'expr': line_expr, 'type': 'segment', 'data': data})
        for data in inf_lines:
            x1, y1 = data['x1'], data['y1']
            x2, y2 = data['x2'], data['y2']
            if x1 == x2:
                line_expr = f'x == {x1}'
            elif y1 == y2:
                line_expr = f'y == {y1}'
            else:
                k = (y2 - y1) / (x2 - x1)
                line_expr = f'({k})*(x - {x1}) + {y1}'
            all_lines.append({'expr': line_expr, 'type': 'infinite', 'data': data})
        expr_list = func_expressions + [l['expr'] for l in all_lines]
        new_intersections = {}
        
        # check the intersections between functions
        for i in range(len(func_expressions)):
            for j in range(i + 1, len(func_expressions)):
                try:
                    expr1 = parse_expr(func_expressions[i], variables={x})
                    expr2 = parse_expr(func_expressions[j], variables={x})
                    eq = Eq(expr1, expr2)
                    sols = solve(eq, x, domain='real')
                    
                    for sol in sols:
                        sol_numeric = N(sol, tolerance=1e-6)
                        if not isinstance(sol_numeric, float):
                            continue
                        
                        sol_decimal = Decimal(str(sol_numeric))
                        x_min = Decimal(str(-X_LIMIT)) - INTERSECTION_TOLERANCE
                        x_max = Decimal(str(X_LIMIT)) + INTERSECTION_TOLERANCE
                        
                        if x_min <= sol_decimal <= x_max:
                            y_numeric = N(expr1.subs(x, sol_numeric), tolerance=1e-6)
                            y_decimal = Decimal(str(float(y_numeric)))
                            
                            key_x = round(sol_decimal / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                            key_y = round(y_decimal / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                            key = (float(key_x), float(key_y))
                            
                            if key not in new_intersections:
                                new_intersections[key] = set()
                            new_intersections[key].update({i, j + len(plots)})
                except Exception:
                    continue
        
        
        line_idx_offset = len(func_expressions)
        for idx, line in enumerate(all_lines):
            line_expr = line['expr']
            try:
                line_eq = parse_expr(line_expr, variables={x})
            except:
                continue
            if 'x ==' in line_expr:
                a = Decimal(line_expr.split('==')[1].strip())
                for func_idx, func_expr_str in enumerate(func_expressions):
                    try:
                        func_expr = parse_expr(func_expr_str, variables={x})
                        y_val = func_expr.subs(x, float(a))
                        y_numeric = N(y_val, tolerance=1e-12)
                        if not isinstance(y_numeric, float):
                            continue
                        y_decimal = Decimal(str(y_numeric))
                        key_x = round(a / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                        key_y = round(y_decimal / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                        key = (float(key_x), float(key_y))
                        if key not in new_intersections:
                            new_intersections[key] = set()
                        new_intersections[key].update({func_idx, idx + line_idx_offset})
                    except Exception:
                        continue
            elif 'y ==' in line_expr:
                b = Decimal(line_expr.split('==')[1].strip())
                for func_idx, func_expr_str in enumerate(func_expressions):
                    try:
                        func_expr = parse_expr(func_expr_str, variables={x})
                        eq = Eq(func_expr, float(b))
                        sols = solve(eq, x, domain='real')
                        for sol in sols:
                            sol_numeric = N(sol, tolerance=1e-12)
                            if not isinstance(sol_numeric, float):
                                continue
                            sol_decimal = Decimal(str(sol_numeric))
                            key_x = round(sol_decimal / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                            key_y = round(b / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                            key = (float(key_x), float(key_y))
                            if key not in new_intersections:
                                new_intersections[key] = set()
                            new_intersections[key].update({func_idx, idx + line_idx_offset})
                    except Exception:
                        continue
            else:
                try:
                    line_lambda = lambdify(x, line_eq, modules=[np, {'math': math}])
                except Exception:
                    continue
                for func_idx, func_expr_str in enumerate(func_expressions):
                    try:
                        func_expr = parse_expr(func_expr_str, variables={x})
                        eq = Eq(func_expr, line_eq)
                        sols = solve(eq, x, domain='real')
                        for sol in sols:
                            sol_numeric = N(sol, tolerance=1e-12)
                            if not isinstance(sol_numeric, float):
                                continue
                            sol_decimal = Decimal(str(sol_numeric))
                            y_numeric = N(line_eq.subs(x, sol_numeric), tolerance=1e-12)
                            y_decimal = Decimal(str(float(y_numeric)))
                            key_x = round(sol_decimal / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                            key_y = round(y_decimal / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                            key = (float(key_x), float(key_y))
                            if key not in new_intersections:
                                new_intersections[key] = set()
                            new_intersections[key].update({func_idx, idx + line_idx_offset})
                    except Exception:
                        continue
        # intersections between lines
        for i in range(len(all_lines)):
            for j in range(i+1, len(all_lines)):
                expr1 = all_lines[i]['expr']
                expr2 = all_lines[j]['expr']
                try:
                    if expr1 == expr2:
                        continue
                    eqs = []
                    if 'x ==' in expr1:
                        a = Decimal(expr1.split('==')[1].strip())
                        eqs.append(('x', a))
                    elif 'y ==' in expr1:
                        b = Decimal(expr1.split('==')[1].strip())
                        eqs.append(('y', b))
                    else:
                        eqs.append(('y', parse_expr(expr1, variables={x})))
                    if 'x ==' in expr2:
                        a2 = Decimal(expr2.split('==')[1].strip())
                        eqs.append(('x', a2))
                    elif 'y ==' in expr2:
                        b2 = Decimal(expr2.split('==')[1].strip())
                        eqs.append(('y', b2))
                    else:
                        eqs.append(('y2', parse_expr(expr2, variables={x})))
                    # get intersections
                    if len(eqs) == 2:
                        if eqs[0][0] == 'x' and eqs[1][0] == 'y':
                            xval = float(eqs[0][1])
                            yval = float(eqs[1][1].subs(x, xval))
                        elif eqs[0][0] == 'y' and eqs[1][0] == 'x':
                            xval = float(eqs[1][1])
                            yval = float(eqs[0][1].subs(x, xval))
                        elif eqs[0][0] == 'x' and eqs[1][0] == 'x':
                            continue
                        elif eqs[0][0] == 'y' and eqs[1][0] == 'y':
                            continue
                        elif eqs[0][0] == 'y' and eqs[1][0] == 'y2':
                            eq = Eq(eqs[0][1], eqs[1][1])
                            sols = solve(eq, x, domain='real')
                            for sol in sols:
                                sol_numeric = N(sol, tolerance=1e-12)
                                if not isinstance(sol_numeric, float):
                                    continue
                                xval = float(sol_numeric)
                                yval = float(eqs[0][1].subs(x, xval))
                                key_x = round(Decimal(str(xval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                                key_y = round(Decimal(str(yval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                                key = (float(key_x), float(key_y))
                                if key not in new_intersections:
                                    new_intersections[key] = set()
                                new_intersections[key].update({i+line_idx_offset, j+line_idx_offset})
                            continue
                        elif eqs[0][0] == 'y2' and eqs[1][0] == 'y':
                            eq = Eq(eqs[0][1], eqs[1][1])
                            sols = solve(eq, x, domain='real')
                            for sol in sols:
                                sol_numeric = N(sol, tolerance=1e-12)
                                if not isinstance(sol_numeric, float):
                                    continue
                                xval = float(sol_numeric)
                                yval = float(eqs[1][1].subs(x, xval))
                                key_x = round(Decimal(str(xval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                                key_y = round(Decimal(str(yval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                                key = (float(key_x), float(key_y))
                                if key not in new_intersections:
                                    new_intersections[key] = set()
                                new_intersections[key].update({i+line_idx_offset, j+line_idx_offset})
                            continue
                        elif eqs[0][0] == 'y2' and eqs[1][0] == 'y2':
                            eq = Eq(eqs[0][1], eqs[1][1])
                            sols = solve(eq, x, domain='real')
                            for sol in sols:
                                sol_numeric = N(sol, tolerance=1e-12)
                                if not isinstance(sol_numeric, float):
                                    continue
                                xval = float(sol_numeric)
                                yval = float(eqs[0][1].subs(x, xval))
                                key_x = round(Decimal(str(xval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                                key_y = round(Decimal(str(yval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                                key = (float(key_x), float(key_y))
                                if key not in new_intersections:
                                    new_intersections[key] = set()
                                new_intersections[key].update({i+line_idx_offset, j+line_idx_offset})
                            continue
                        else:
                            continue
                        key_x = round(Decimal(str(xval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                        key_y = round(Decimal(str(yval)) / INTERSECTION_TOLERANCE) * INTERSECTION_TOLERANCE
                        key = (float(key_x), float(key_y))
                        if key not in new_intersections:
                            new_intersections[key] = set()
                        new_intersections[key].update({i+line_idx_offset, j+line_idx_offset})
                except Exception:
                    continue
        intersection_points = {
            key: list(indices) for key, indices in new_intersections.items() 
            if len(indices) >= 2 and -X_LIMIT <= key[0] <= X_LIMIT
        }

    def add_function(func_str):
        """add the functions into the graph"""
        try:
            func = parse_expression(func_str)
            if func is None:
                return

            fig = plt.gcf()
            ax = plt.gca()
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width_px = int(bbox.width * fig.dpi)
            n_points = min(max(width_px * 2, 2000), 1000000) # how many points drawn 
            x_np = np.linspace(-X_LIMIT, X_LIMIT, n_points)
            original_length = len(x_np)

            # dealing with special domains
            if 'sqrt' in func_str.lower():
                valid_mask = x_np >= 0
                x_np = x_np[valid_mask]

            # calculating y values and filter the i values
            y_np = func(x_np)
            valid_mask = np.isfinite(y_np)
            x_np = x_np[valid_mask]
            y_np = y_np[valid_mask]

            # check if there are any i points
            if original_length > len(x_np):
                has_imaginary.add(func_str)

            if len(x_np) == 0:
                print("Function has no valid real values in the domain")
                return

            # updating the area of the axis
            update_global_limits(x_np, y_np)

            # draw the functions
            color = colors[len(plots) % len(colors)]
            sizes = np.full(len(x_np), 3)  
            jitter = (np.random.rand(len(x_np)) - 0.5) * (X_LIMIT * 0.001)
            x_jitter = x_np + jitter
            scatter = plt.scatter(
                x_jitter, y_np, s=sizes, color=color,
                edgecolor=color, alpha=0.2, label=func_str
            )

            # storing the data of the functions
            plots[scatter] = {
                'x': x_jitter, 'y': y_np, 'expression': func_str,
                'color': color, 'sizes': sizes
            }

            # showing the tip of i existing
            show_imaginary_warning()

            plt.legend()
            calculate_intersections()
            plt.draw()

        except Exception as e:
            print(f"Error adding function: {str(e)}")
            return

    def show_imaginary_warning():
        """i warning"""
        # deleting exsisted warning
        for text in ax.texts:
            if "imaginary values" in text.get_text():
                text.remove()
        
        # adding new warning
        if has_imaginary:
            ax.text(0.01, 0.99, 
                    "Some functions have imaginary values for certain inputs", 
                    transform=ax.transAxes, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=8)

    def add_line(line_str):
        """adding new lines in graph"""
        pattern = r'draw line segment (\w+) \(([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\), (\w+) \(([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\) as line segment (\w+)(\w+)$'
        match = re.match(pattern, line_str)

        if not match:
            print("Invalid line format. Use: 'draw line segment A (x1, y1), B (x2, y2) as line segment AB'")
            return

        try:
            name1, x1_str, y1_str, name2, x2_str, y2_str, seg_name1, seg_name2 = match.groups()
            x1, y1 = float(x1_str), float(y1_str)
            x2, y2 = float(x2_str), float(y2_str)

            t = np.linspace(0, 1, 100)
            x_np = x1 + t * (x2 - x1)
            y_np = y1 + t * (y2 - y1)

            update_global_limits(x_np, y_np)

            color = colors[(len(plots) + len(lines)) % len(colors)]
            sizes = np.full(len(x_np), 1)
            scatter = plt.scatter(
                x_np, y_np, s=sizes, color=color,
                edgecolor=color, alpha=1, label=f"line segment {seg_name1}{seg_name2}"
            )

            lines.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'x': x_np, 'y': y_np, 'color': color,
                'sizes': sizes, 'name1': name1, 'name2': name2,
                'scatter': scatter,
                'seg_name': f'{seg_name1}{seg_name2}'
            })

            plt.legend()
            calculate_intersections()
            plt.draw()

        except Exception as e:
            print(f"Error adding line: {str(e)}")
            return

    def on_hover(event):
        """dealing with the mouse hover point"""
        # deleting exsisting hover points
        for point in hover_points:
            point.remove()
        hover_points.clear()
        
        if event.inaxes != ax:
            plt.title('Geometry Drawer', y=1.02)
            return
            
        # replacing the objects' appearance
        for scatter in plots:
            scatter.set_sizes(plots[scatter]['sizes'])
            scatter.set_facecolor(plots[scatter]['color'])
        for data in lines:
            data['scatter'].set_sizes(data['sizes'])
            data['scatter'].set_facecolor(data['color'])
            
        if event.xdata is None:
            plt.title('Geometry Drawer', y=1.02)
            return
            
        # finding the nearest point
        closest_points = []
        
        # checking the graph
        for scatter, data in plots.items():
            x_domain = data['x']
            if not x_domain.size:
                continue
                
            distances = np.hypot(x_domain - event.xdata, data['y'] - event.ydata)
            min_idx = np.argmin(distances)
            closest_x = x_domain[min_idx]
            closest_y = data['y'][min_idx]
            
            if distances[min_idx] < 10:
                is_intersection = (round(closest_x, 4), round(closest_y, 4)) in intersection_points
                point = ax.scatter(
                    closest_x, closest_y, s=50, 
                    color='black' if is_intersection else data['color'],
                    edgecolor='black', zorder=10
                )
                hover_points.append(point)
                closest_points.append((closest_x, closest_y, data['expression']))
        
        # checking the lines
        for data in lines:
            x_domain = data['x']
            y_domain = data['y']
            if not x_domain.size:
                continue
            distances = np.hypot(x_domain - event.xdata, y_domain - event.ydata)
            min_idx = np.argmin(distances)
            closest_x = x_domain[min_idx]
            closest_y = y_domain[min_idx]
            if distances[min_idx] < 10:
                is_intersection = (round(closest_x, 4), round(closest_y, 4)) in intersection_points
                point = ax.scatter(
                    closest_x, closest_y, s=50, 
                    color='black' if is_intersection else data['color'],
                    edgecolor='black', zorder=10
                )
                hover_points.append(point)
                closest_points.append((closest_x, closest_y, f"Line segment {data['name1']}-{data['name2']}"))
        for data in inf_lines:
            x_domain = data['x']
            y_domain = data['y']
            if not x_domain.size:
                continue
            distances = np.hypot(x_domain - event.xdata, y_domain - event.ydata)
            min_idx = np.argmin(distances)
            closest_x = x_domain[min_idx]
            closest_y = y_domain[min_idx]
            if distances[min_idx] < 10:
                is_intersection = (round(closest_x, 4), round(closest_y, 4)) in intersection_points
                point = ax.scatter(
                    closest_x, closest_y, s=50, 
                    color='black' if is_intersection else data['color'],
                    edgecolor='black', zorder=10
                )
                hover_points.append(point)
                closest_points.append((closest_x, closest_y, f"Line {data['name1']}-{data['name2']}"))
        
        # updating the title
        if closest_points:
            closest_points.sort(key=lambda p: np.hypot(p[0]-event.xdata, p[1]-event.ydata))
            cx, cy, expr = closest_points[0]
            plt.title(f"Point ({cx:.2f}, {cy:.2f}) on {expr}", y=1.02)
        else:
            plt.title('Geometry Drawer', y=1.02)
        
        # light up the intersection point
        for (x_val, y_val), indices in intersection_points.items():
            obj_list = list(plots.items()) + [(data['scatter'], data) for data in lines] + [(data['scatter'], data) for data in inf_lines]
            for obj_idx, (scatter, data) in enumerate(obj_list):
                x_domain = data['x']
                y_domain = data['y']
                distances = np.hypot(x_domain - x_val, y_domain - y_val)
                mask = distances < float(INTERSECTION_TOLERANCE * 2)
                if np.any(mask):
                    min_idx = np.argmin(distances[mask])
                    if 'sizes' in data and obj_idx < len(plots):
                        data['sizes'][mask][min_idx] = 12
                    scatter.set_markeredgecolor('black')
                    scatter.set_markerfacecolor('black')
        
        # update the size and draw again
        for scatter in plots:
            scatter.set_sizes(plots[scatter]['sizes'])
        for data in lines:
            data['scatter'].set_sizes(data['sizes'])
        plt.draw()

    # initing the area of the graph
    ax.set_xlim(-X_LIMIT, X_LIMIT)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    
    # cobine the graph
    plt.connect('motion_notify_event', on_hover)
    
    # input
    while True:
        try:
            user_input = input("Please input (type 'exit' to stop entering and draw the graph): ")
            if user_input.lower() == 'exit':
                break
                
            if user_input.lower().startswith('draw line segment'):
                add_line(user_input)
                continue
            if user_input.lower().startswith('draw line'):
                add_infinite_line(user_input)
                continue
                
            # validate the inputs
            if not re.match(r'^[\w\s()+\-*/√πe.,lnlog]+$', user_input):
                print("Invalid input. Allowed: 'draw line' commands or expressions with +, -, *, /, pi, sqrt, log, ln, **, (), .")
                continue
                
            # adding functions
            safe_expr = user_input.replace('√', 'sqrt')
            add_function(safe_expr)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    plt.show()
create_interactive_plot()