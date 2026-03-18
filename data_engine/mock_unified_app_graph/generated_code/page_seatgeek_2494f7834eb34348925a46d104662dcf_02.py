# page_id: page_seatgeek_2494f7834eb34348925a46d104662dcf_02
# screenshot: 2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5.png
# step_index: 2/9
# task: Open SeatGeek. Search for "Book of Mormon". Add the show to favorite. Select date April 26. Set the ticket number to 2 and proceed. What is the lowest price for each ticket?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the page

W, H = canvas.size

# Colors
bg_white = (255, 255, 255)
status_bg = (236, 236, 236)      # light grey status bar
search_bg = (243, 244, 246)      # very light grey search field
divider = (230, 230, 230)        # thin divider lines
card_outline = (240, 240, 240)   # subtle card outline
shadow_line = (235, 235, 235)    # subtle shadow
nav_top_line = (230, 230, 230)

# Fill overall background (canvas starts white, but be explicit)
draw.rectangle([0, 0, W, H], fill=bg_white)

# 1) Status bar area (top)
status_h = 80
draw.rectangle([0, 0, W, status_h], fill=status_bg)
# bottom divider of status bar
draw.line([(0, status_h), (W, status_h)], fill=divider, width=1)

# 2) Search bar (rounded) near the top. Keep inner content (icons/text) omitted.
search_x0 = 48
search_x1 = W - 48
search_y0 = 120
search_h = 144
search_y1 = search_y0 + search_h
draw.rounded_rectangle([search_x0, search_y0, search_x1, search_y1],
                       radius=36, fill=search_bg, outline=card_outline, width=1)

# subtle inner divider under search area
line_y = search_y1 + 36
draw.line([(search_x0, line_y), (search_x1, line_y)], fill=divider, width=1)

# 3) Recent searches / list group card background (subtle rounded card area)
# This provides the white card/background behind the list items (no icons/text drawn)
list_card_x0 = 32
list_card_x1 = W - 32
list_card_y0 = line_y + 24
# Extend down to include several list items (approximate)
list_card_y1 = 1500
draw.rounded_rectangle([list_card_x0, list_card_y0, list_card_x1, list_card_y1],
                       radius=12, fill=bg_white, outline=card_outline, width=1)

# 4) Separators between list items (approximate positions based on detected sizes)
# Detected list items are roughly 168px tall; draw separators at those intervals.
first_item_top = 468  # approximate top of first list item row
rows = 8
row_h = 168
sep_x0 = list_card_x0 + 8
sep_x1 = list_card_x1 - 8
for i in range(rows):
    y = first_item_top + i * row_h
    # draw a faint separator line (do not draw over the whole width to mimic UI padding)
    draw.line([(sep_x0, y), (sep_x1, y)], fill=divider, width=1)

# 5) Section divider between Recent searches and Suggestions
suggestions_divider_y = 1408
draw.line([(list_card_x0, suggestions_divider_y), (list_card_x1, suggestions_divider_y)],
          fill=divider, width=2)

# 6) Suggestions card area (rounded card behind suggestion items)
suggest_x0 = 32
suggest_x1 = W - 32
suggest_y0 = suggestions_divider_y + 24
suggest_y1 = 2020
draw.rounded_rectangle([suggest_x0, suggest_y0, suggest_x1, suggest_y1],
                       radius=12, fill=bg_white, outline=card_outline, width=1)

# separators within suggestions (approx positions of three suggestion rows)
suggest_first_row_top = 1520
for i in range(3):
    y = suggest_first_row_top + i * 168
    draw.line([(sep_x0, y), (sep_x1, y)], fill=divider, width=1)

# 7) Large horizontal rule further down to visually separate content from whitespace
draw.line([(24, suggest_y1 + 40), (W - 24, suggest_y1 + 40)], fill=shadow_line, width=1)

# 8) Bottom navigation bar area (background + top divider/shadow)
nav_top = 2792
draw.rectangle([0, nav_top, W, H], fill=bg_white)
draw.line([(0, nav_top), (W, nav_top)], fill=nav_top_line, width=2)

# 9) Very subtle top shadow for list card (to separate from background)
shadow_y = list_card_y0 - 6
draw.line([(list_card_x0 + 6, shadow_y), (list_card_x1 - 6, shadow_y)],
          fill=shadow_line, width=1)

# End of structural drawing

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/00_icon_The_Lion_King.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["The_Lion_King"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/01_icon_Boston_Celtics.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 807), _c1)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/02_icon_Recent_searches.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 471), _c2)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 48, 70)
    canvas.paste(_c3, (1153, 0), _c3)
except Exception:
    pass
layout["icon_3"] = [1153, 0, 1201, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/04_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 639), _c4)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/05_icon_Tracking.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (864, 2792), _c5)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 97, 69)
    canvas.paste(_c7, (1216, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1216, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/08_icon_The_Phantom_of_the_Opera.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 807), _c8)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 66, 61)
    canvas.paste(_c9, (242, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [242, 3, 308, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/10_icon_Cirque_du_Soleil_The_Beatles.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 471), _c10)
except Exception:
    pass
layout["Cirque_du_Soleil:_The_Bea"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/11_icon_Wicked.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 975), _c11)
except Exception:
    pass
layout["Wicked"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/12_icon_Tickets.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (576, 2792), _c12)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/13_icon_Just_Announced_by_My_Performers.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 1688), _c13)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/14_icon_The_Phantom_of_the_Opera.png
try:
    _c14 = get_crop(14, 1440, 168)
    canvas.paste(_c14, (0, 975), _c14)
except Exception:
    pass
layout["The_Phantom_of_the_Opera"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/15_icon_Clear.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 120), _c15)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/16_icon_6.49_my.png
try:
    _c16 = get_crop(16, 168, 144)
    canvas.paste(_c16, (48, 120), _c16)
except Exception:
    pass
layout["6.49_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/17_icon_6.49_my.png
try:
    _c17 = get_crop(17, 47, 63)
    canvas.paste(_c17, (186, 1), _c17)
except Exception:
    pass
layout["6.49_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 52, 68)
    canvas.paste(_c18, (1319, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/19_icon_Account.png
try:
    _c19 = get_crop(19, 288, 168)
    canvas.paste(_c19, (1152, 2792), _c19)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/20_icon_Events_by_My_Performers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 1520), _c20)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/21_icon_6.49_my.png
try:
    _c21 = get_crop(21, 58, 65)
    canvas.paste(_c21, (113, 0), _c21)
except Exception:
    pass
layout["6.49_my"] = [113, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/22_icon_Search.png
try:
    _c22 = get_crop(22, 288, 162)
    canvas.paste(_c22, (288, 2792), _c22)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/23_icon_Boston_Celtics.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1143), _c23)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/25_icon_Wicked.png
try:
    _c25 = get_crop(25, 1440, 168)
    canvas.paste(_c25, (0, 1143), _c25)
except Exception:
    pass
layout["Wicked"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/26_icon_Boston_Celtics.png
try:
    _c26 = get_crop(26, 1440, 168)
    canvas.paste(_c26, (0, 975), _c26)
except Exception:
    pass
layout["Boston_Celtics"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/27_icon_icon_27.png
try:
    _c27 = get_crop(27, 48, 58)
    canvas.paste(_c27, (383, 5), _c27)
except Exception:
    pass
layout["icon_27"] = [383, 5, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/28_icon_Just_Announced_by_My_Performers.png
try:
    _c28 = get_crop(28, 1440, 168)
    canvas.paste(_c28, (0, 1856), _c28)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/29_icon_Search.png
try:
    _c29 = get_crop(29, 288, 162)
    canvas.paste(_c29, (288, 2792), _c29)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/30_text_Recent_searches.png
try:
    _c30 = get_crop(30, 168, 144)
    canvas.paste(_c30, (48, 120), _c30)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2494f7834eb34348925a46d104662dcf/step_02_2024_4_22_18_48_2494f7834eb34348925a46d104662dcf-5/31_text_Suggestions.png
try:
    _c31 = get_crop(31, 331, 74)
    canvas.paste(_c31, (40, 1423), _c31)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
