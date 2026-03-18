# page_id: page_seatgeek_86974bc0508841cfb7a0668793029b53_02
# screenshot: 2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5.png
# step_index: 2/5
# task: Open SeatGeek. Search for the "Ed Sheeran" concert. Check the next upcoming event. When and where is the concert?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw structural UI background for the provided canvas using PIL draw (canvas, draw available)

# Colors
bg_color = (255, 255, 255)          # main white background
status_bar_color = (245, 246, 247)  # very light gray for status area
search_bg = (242, 243, 244)         # search field background
divider = (233, 233, 233)           # thin separators
muted_bg = (250, 250, 250)          # subtle card background
bottom_border = (237, 237, 237)     # nav top border

W, H = canvas.size

# Fill overall background (canvas starts white but ensure consistent color)
draw.rectangle([0, 0, W, H], fill=bg_color)

# Status bar area (top)
status_h = 120
draw.rectangle([0, 0, W, status_h], fill=status_bar_color)

# Search bar background (rounded rectangle)
search_left = 48
search_right = W - 48
search_top = 120
search_height = 144
search_bottom = search_top + search_height
search_radius = 28
draw.rounded_rectangle([search_left, search_top, search_right, search_bottom],
                       radius=search_radius, fill=search_bg)

# Subtle divider under header / search region
draw.line([(48, search_bottom + 24), (W - 48, search_bottom + 24)], fill=divider, width=1)

# "Recent searches" region separators (approximate positions matching detected list spacing)
# Based on detected rows spaced ~168px apart starting near y=471
# We'll draw separators at the bottoms of each list row so pasted items (icons/text) remain on top.
list_start_y = 471
row_height = 168
num_rows = 5
for i in range(1, num_rows):
    y = list_start_y + i * row_height
    # draw from left content margin to right content margin
    draw.line([(48, y), (W - 48, y)], fill=divider, width=1)

# Additional light divider between Recent Searches and Suggestions
sep_between = list_start_y + num_rows * row_height + 24
draw.line([(48, sep_between), (W - 48, sep_between)], fill=divider, width=1)

# Suggestions group background (subtle rounded card behind items area)
suggestions_top = sep_between + 28
suggestions_bottom = suggestions_top + 360
card_left = 40
card_right = W - 40
card_radius = 18
draw.rounded_rectangle([card_left, suggestions_top, card_right, suggestions_bottom],
                       radius=card_radius, fill=muted_bg, outline=None)

# Inside the suggestions card, draw separators for each suggestion row (three items)
# Position rows with comfortable padding
card_pad_left = card_left + 40
card_pad_right = card_right - 40
first_row_y = suggestions_top + 36
row_spacing = 92
for i in range(1, 3):
    y = first_row_y + i * row_spacing
    draw.line([(card_pad_left, y), (card_pad_right, y)], fill=divider, width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
nav_bottom = H
draw.rectangle([0, nav_top, W, nav_bottom], fill=bg_color)
draw.line([(0, nav_top), (W, nav_top)], fill=bottom_border, width=2)

# Slight left and right edge guides for consistent content margins (very light)
edge_line_color = (248, 248, 248)
draw.line([(48, status_h), (48, H - 200)], fill=edge_line_color, width=1)
draw.line([(W - 48, status_h), (W - 48, H - 200)], fill=edge_line_color, width=1)

# Subtle emphasis bar under search field (thin shadow-like stripe)
shadow_y = search_bottom + 6
draw.line([(search_left + 6, shadow_y), (search_right - 6, shadow_y)], fill=(245,245,245), width=1)

# Final accent: very light rounded card behind top-of-list header area to separate from search
header_card_top = search_bottom + 36
header_card_bottom = header_card_top + 72
draw.rounded_rectangle([search_left, header_card_top, search_right, header_card_bottom],
                       radius=12, fill=bg_color, outline=divider)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/00_icon_Justin_Bieber.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["Justin_Bieber"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/01_icon_Justin_Bieber.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 639), _c1)
except Exception:
    pass
layout["Justin_Bieber"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/02_icon_icon_2.png
try:
    _c2 = get_crop(2, 49, 69)
    canvas.paste(_c2, (1152, 0), _c2)
except Exception:
    pass
layout["icon_2"] = [1152, 0, 1201, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 64, 65)
    canvas.paste(_c3, (242, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [242, 2, 306, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/04_icon_Tracking.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (864, 2792), _c4)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/05_icon_icon_5.png
try:
    _c5 = get_crop(5, 98, 69)
    canvas.paste(_c5, (1215, 0), _c5)
except Exception:
    pass
layout["icon_5"] = [1215, 0, 1313, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/06_icon_Metropolitan_Opera.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 471), _c6)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/07_icon_Browse.png
try:
    _c7 = get_crop(7, 288, 168)
    canvas.paste(_c7, (0, 2792), _c7)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/08_icon_Just_Announced_by_My_Performers.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 1688), _c8)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/10_icon_8.00_my.png
try:
    _c10 = get_crop(10, 168, 144)
    canvas.paste(_c10, (48, 120), _c10)
except Exception:
    pass
layout["8.00_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/11_icon_Clear.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 120), _c11)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 52, 68)
    canvas.paste(_c12, (1319, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1319, 0, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/13_icon_8.00_my.png
try:
    _c13 = get_crop(13, 47, 64)
    canvas.paste(_c13, (186, 1), _c13)
except Exception:
    pass
layout["8.00_my"] = [186, 1, 233, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 62, 64)
    canvas.paste(_c15, (313, 2), _c15)
except Exception:
    pass
layout["icon_15"] = [313, 2, 375, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/16_icon_Events_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1520), _c16)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/17_icon_Los_Angeles_Lakers.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 639), _c17)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/18_icon_Madison_Square_Garden.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1143), _c18)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/19_icon_Los_Angeles_Lakers.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 807), _c19)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/20_icon_Madison_Square_Garden.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 975), _c20)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/22_icon_Just_Announced_by_My_Performers.png
try:
    _c22 = get_crop(22, 1440, 168)
    canvas.paste(_c22, (0, 1856), _c22)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/23_icon_8.00_my.png
try:
    _c23 = get_crop(23, 58, 65)
    canvas.paste(_c23, (113, 0), _c23)
except Exception:
    pass
layout["8.00_my"] = [113, 0, 171, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event,_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/25_text_Recent_searches.png
try:
    _c25 = get_crop(25, 168, 144)
    canvas.paste(_c25, (48, 120), _c25)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_02_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-5/26_text_Suggestions.png
try:
    _c26 = get_crop(26, 331, 74)
    canvas.paste(_c26, (40, 1423), _c26)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
