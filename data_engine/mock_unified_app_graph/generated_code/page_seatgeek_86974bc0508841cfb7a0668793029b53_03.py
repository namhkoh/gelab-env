# page_id: page_seatgeek_86974bc0508841cfb7a0668793029b53_03
# screenshot: 2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6.png
# step_index: 3/5
# task: Open SeatGeek. Search for the "Ed Sheeran" concert. Check the next upcoming event. When and where is the concert?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background and structural UI elements for the mobile page.
# Assumes variables provided: canvas (PIL Image), draw (ImageDraw)

w, h = canvas.size

# Colors
BG = (255, 255, 255)
STATUS_BG = (248, 248, 249)      # very light status bar background
SEARCH_BG = (250, 250, 250)      # search field background
SEARCH_BORDER = (236, 236, 236)  # subtle border for search field
DIVIDER = (235, 235, 235)        # thin dividers between sections
CARD_BG = (252, 252, 252)        # soft card background for suggestion block
BOTTOM_BORDER = (230, 230, 230)  # top line of bottom navigation

# Fill canvas background
draw.rectangle([0, 0, w, h], fill=BG)

# Status bar area at top (~56px)
status_height = 56
draw.rectangle([0, 0, w, status_height], fill=STATUS_BG)

# Header area spacer (between status bar and search) - keep same background as page
header_bottom = status_height + 64  # leaves room before search field
draw.rectangle([0, status_height, w, header_bottom], fill=BG)

# Search bar rounded background
# Use coordinates aligned with detected search region (top at y=120, height ~144)
search_top = 120
search_height = 144
search_left = 40
search_right = w - 40
search_bottom = search_top + search_height
search_radius = 28
draw.rounded_rectangle([search_left, search_top, search_right, search_bottom],
                       radius=search_radius, fill=SEARCH_BG, outline=SEARCH_BORDER, width=1)

# Subtle shadow line below search area (light divider)
search_divider_y = search_bottom + 36
draw.line([(24, search_divider_y), (w - 24, search_divider_y)], fill=DIVIDER, width=1)

# Divider between "Recent searches" list and "Suggestions"
# Based on detected content positions: last recent-search item ends near y ~1311
recent_items_bottom = 1311
group_divider_y = recent_items_bottom + 9
draw.line([(24, group_divider_y), (w - 24, group_divider_y)], fill=DIVIDER, width=1)

# Suggestions block background (rounded card behind suggestion items)
# Keep it subtle and light so pasted icons/text sit on top
suggestions_top = group_divider_y + 40
suggestions_bottom = suggestions_top + 420
card_left = 24
card_right = w - 24
card_radius = 16
draw.rounded_rectangle([card_left, suggestions_top, card_right, suggestions_bottom],
                       radius=card_radius, fill=CARD_BG, outline=None)

# Another faint divider to visually separate sections lower on the page (near where other sections start)
lower_section_div_y = suggestions_bottom + 40
draw.line([(24, lower_section_div_y), (w - 24, lower_section_div_y)], fill=DIVIDER, width=1)

# Bottom navigation bar background and top border
bottom_nav_top = 2792
draw.rectangle([0, bottom_nav_top, w, h], fill=BG)
draw.line([(0, bottom_nav_top), (w, bottom_nav_top)], fill=BOTTOM_BORDER, width=1)

# Add a very subtle top shadow for the bottom nav (single pixel for separation)
shadow_y = bottom_nav_top + 1
draw.line([(0, shadow_y), (w, shadow_y)], fill=(244,244,244), width=1)

# Final subtle accents: a faint left and right padding lines near the search area to match UI rhythm
pad_line_y1 = search_top + 24
pad_line_y2 = search_bottom - 24
draw.line([(24, pad_line_y1), (24, pad_line_y2)], fill=BG, width=1)
draw.line([(w - 24, pad_line_y1), (w - 24, pad_line_y2)], fill=BG, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/00_icon_Justin_Bieber.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 807), _c0)
except Exception:
    pass
layout["Justin_Bieber"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/01_icon_Metropolitan_Opera.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["Metropolitan_Opera"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/02_icon_Justin_Bieber.png
try:
    _c2 = get_crop(2, 1440, 168)
    canvas.paste(_c2, (0, 639), _c2)
except Exception:
    pass
layout["Justin_Bieber"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/03_icon_8.00_my.png
try:
    _c3 = get_crop(3, 168, 144)
    canvas.paste(_c3, (48, 120), _c3)
except Exception:
    pass
layout["8.00_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 47, 70)
    canvas.paste(_c4, (1153, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/05_icon_Tracking.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (864, 2792), _c5)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/06_icon_Browse.png
try:
    _c6 = get_crop(6, 288, 168)
    canvas.paste(_c6, (0, 2792), _c6)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/07_icon_Just_Announced_by_My_Performers.png
try:
    _c7 = get_crop(7, 1440, 168)
    canvas.paste(_c7, (0, 1688), _c7)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 62, 64)
    canvas.paste(_c8, (242, 2), _c8)
except Exception:
    pass
layout["icon_8"] = [242, 2, 304, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 95, 68)
    canvas.paste(_c10, (1217, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1217, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/11_icon_Clear.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 120), _c11)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/12_icon_Events_by_My_Performers.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 1520), _c12)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/13_icon_Los_Angeles_Lakers.png
try:
    _c13 = get_crop(13, 1440, 168)
    canvas.paste(_c13, (0, 639), _c13)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 53, 68)
    canvas.paste(_c15, (1319, 0), _c15)
except Exception:
    pass
layout["icon_15"] = [1319, 0, 1372, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 59, 64)
    canvas.paste(_c16, (313, 2), _c16)
except Exception:
    pass
layout["icon_16"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/17_icon_Madison_Square_Garden.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1143), _c17)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/18_icon_8.00_my.png
try:
    _c18 = get_crop(18, 47, 63)
    canvas.paste(_c18, (186, 1), _c18)
except Exception:
    pass
layout["8.00_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/19_icon_Madison_Square_Garden.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 975), _c19)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/20_icon_Los_Angeles_Lakers.png
try:
    _c20 = get_crop(20, 1440, 168)
    canvas.paste(_c20, (0, 807), _c20)
except Exception:
    pass
layout["Los_Angeles_Lakers"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/22_icon_Seattle_Mariners.png
try:
    _c22 = get_crop(22, 141, 134)
    canvas.paste(_c22, (41, 996), _c22)
except Exception:
    pass
layout["Seattle_Mariners"] = [41, 996, 182, 1130]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/23_icon_Just_Announced_by_My_Performers.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1856), _c23)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/24_icon_Performer_event_or_venue.png
try:
    _c24 = get_crop(24, 1032, 144)
    canvas.paste(_c24, (216, 120), _c24)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/25_icon_Search.png
try:
    _c25 = get_crop(25, 288, 162)
    canvas.paste(_c25, (288, 2792), _c25)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/26_text_8.00_my.png
try:
    _c26 = get_crop(26, 153, 52)
    canvas.paste(_c26, (19, 9), _c26)
except Exception:
    pass
layout["8.00_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/27_text_Recent_searches.png
try:
    _c27 = get_crop(27, 168, 144)
    canvas.paste(_c27, (48, 120), _c27)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_03_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-6/28_text_Suggestions.png
try:
    _c28 = get_crop(28, 331, 74)
    canvas.paste(_c28, (40, 1423), _c28)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
