# page_id: page_seatgeek_42f99a380cf348fea73bd58c0e4ec8db_01
# screenshot: 2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4.png
# step_index: 1/14
# task: Open SeatGeek and search for the broadway show "lion king" on March 22. I need 3 tickets at average price less than 500 USD. Find the best seats and record the total price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background and structural UI elements for SeatGeek-like page
# Uses provided: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Define common colors
BG = (250, 250, 250)            # overall app background (very light gray)
STATUS_BG = (246, 246, 247)     # status bar background
HEADER_BG = (255, 255, 255)     # header area (white)
DIVIDER = (230, 230, 230)       # subtle divider lines
HERO_BG = (20, 20, 20)          # dark hero card background
CARD_BG = (255, 255, 255)       # card background (white)
SUBCARD_BG = (34, 40, 49)       # small card placeholder (dark)
NAV_BG = (255, 255, 255)        # bottom nav background
SHADOW = (0, 0, 0, 24)          # used for subtle shadow (if needed)

W, H = canvas.size

# Fill the overall background
draw.rectangle([(0, 0), (W, H)], fill=BG)

# Status bar at top (~80px tall)
status_h = 80
draw.rectangle([(0, 0), (W, status_h)], fill=STATUS_BG)
# subtle bottom line for status bar
draw.line([(0, status_h - 1), (W, status_h - 1)], fill=DIVIDER, width=1)

# Header / toolbar area below status bar (~140px)
header_top = status_h
header_h = 140
draw.rectangle([(0, header_top), (W, header_top + header_h)], fill=HEADER_BG)
# header bottom divider
draw.line([(48, header_top + header_h - 1), (W - 48, header_top + header_h - 1)], fill=DIVIDER, width=1)

# Large hero/promotional card (rounded) below header
hero_x0 = 48
hero_x1 = W - 48
hero_top = header_top + header_h + 12   # small gap after header
hero_h = 740
hero_bottom = hero_top + hero_h
hero_radius = 36
draw.rounded_rectangle([(hero_x0, hero_top), (hero_x1, hero_bottom)], radius=hero_radius, fill=HERO_BG)

# Add a faint inner rounded rect to mimic subtle border of hero card
inner_margin = 8
draw.rounded_rectangle(
    [(hero_x0 + inner_margin, hero_top + inner_margin), (hero_x1 - inner_margin, hero_bottom - inner_margin)],
    radius=max(0, hero_radius - inner_margin),
    outline=(40, 40, 40),
    width=1
)

# Divider below hero to separate sections
section_div_y = hero_bottom + 24
draw.line([(48, section_div_y), (W - 48, section_div_y)], fill=DIVIDER, width=1)

# Trending events section area (just structure/background)
trending_top = section_div_y + 24
trending_h = 360
trending_bottom = trending_top + trending_h
# Keep background same as main (white), but draw a faint card-like area for the list
list_x0 = 0 + 48
list_x1 = W - 48
list_radius = 6
draw.rounded_rectangle([(list_x0, trending_top), (list_x1, trending_bottom)], radius=list_radius, fill=CARD_BG)
# Divider lines between trending items (three items)
item_height = 110
for i in range(1, 3):
    y = trending_top + i * item_height
    draw.line([(list_x0 + 16, y), (list_x1 - 16, y)], fill=DIVIDER, width=1)

# Separator before Recently viewed area
recent_top = trending_bottom + 40
draw.line([(48, recent_top - 16), (W - 48, recent_top - 16)], fill=DIVIDER, width=1)

# Recently viewed horizontal cards (three placeholders)
card_w = 392
card_h = 240
card_gap = 24
start_x = 48
card_y = recent_top
# Draw 3 card placeholders (rounded dark rectangles) that will be behind pasted thumbnails
for i in range(3):
    x0 = start_x + i * (card_w + card_gap)
    x1 = x0 + card_w
    # keep cards within canvas
    if x1 > W - 48:
        x1 = W - 48
    draw.rounded_rectangle([(x0, card_y), (x1, card_y + card_h)], radius=16, fill=SUBCARD_BG)
    # Add a faint inner highlight top-left to emulate overlay area (no text)
    highlight_h = 44
    draw.rectangle([(x0 + 16, card_y + 16), (x1 - 16, card_y + 16 + highlight_h)], fill=(28, 34, 45))

# Divider above bottom navigation
nav_h = 120
nav_top = H - nav_h
draw.line([(0, nav_top), (W, nav_top)], fill=DIVIDER, width=1)
# Bottom navigation background
draw.rectangle([(0, nav_top), (W, H)], fill=NAV_BG)
# Slight top shadow for nav (thin translucent line)
draw.line([(0, nav_top + 1), (W, nav_top + 1)], fill=(0, 0, 0, 12), width=1)

# Additional subtle separators for visual grouping
# faint horizontal line below header right area (to match app chrome)
draw.line([(48, header_top + header_h + 6), (W - 48, header_top + header_h + 6)], fill=(245, 245, 245), width=1)

# End of structural drawing - leave all icons/text areas empty for overlays

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/00_icon_JUSTIN_peck.png
try:
    _c0 = get_crop(0, 462, 519)
    canvas.paste(_c0, (546, 2382), _c0)
except Exception:
    pass
layout["JUSTIN_peck"] = [546, 2382, 1008, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/01_icon_S165.png
try:
    _c1 = get_crop(1, 462, 519)
    canvas.paste(_c1, (48, 2382), _c1)
except Exception:
    pass
layout["S165+"] = [48, 2382, 510, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/02_icon_S216.png
try:
    _c2 = get_crop(2, 396, 519)
    canvas.paste(_c2, (1044, 2382), _c2)
except Exception:
    pass
layout["S216+"] = [1044, 2382, 1440, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/03_icon_NCAA_M_Basketball_Brooklyn.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 1667), _c3)
except Exception:
    pass
layout["NCAA_M_Basketball_Brookly"] = [0, 1667, 1309, 1903]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/04_icon_St._James_Theatre.png
try:
    _c4 = get_crop(4, 1309, 236)
    canvas.paste(_c4, (0, 1431), _c4)
except Exception:
    pass
layout["St._James_Theatre"] = [0, 1431, 1309, 1667]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/05_icon_View_all.png
try:
    _c5 = get_crop(5, 100, 148)
    canvas.paste(_c5, (1340, 1949), _c5)
except Exception:
    pass
layout["View_all"] = [1340, 1949, 1440, 2097]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/06_icon_View_all.png
try:
    _c6 = get_crop(6, 102, 143)
    canvas.paste(_c6, (1338, 1480), _c6)
except Exception:
    pass
layout["View_all"] = [1338, 1480, 1440, 1623]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/07_icon_840.png
try:
    _c7 = get_crop(7, 144, 240)
    canvas.paste(_c7, (1260, 72), _c7)
except Exception:
    pass
layout["840"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/08_icon_GEK.png
try:
    _c8 = get_crop(8, 51, 57)
    canvas.paste(_c8, (250, 4), _c8)
except Exception:
    pass
layout["GEK"] = [250, 4, 301, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 102, 146)
    canvas.paste(_c9, (1338, 1711), _c9)
except Exception:
    pass
layout["icon_9"] = [1338, 1711, 1440, 1857]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/10_icon_Wy.png
try:
    _c10 = get_crop(10, 52, 61)
    canvas.paste(_c10, (116, 3), _c10)
except Exception:
    pass
layout["Wy"] = [116, 3, 168, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/11_icon_Hamilton.png
try:
    _c11 = get_crop(11, 288, 162)
    canvas.paste(_c11, (0, 2792), _c11)
except Exception:
    pass
layout["Hamilton"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/12_icon_840.png
try:
    _c12 = get_crop(12, 97, 62)
    canvas.paste(_c12, (1217, 1), _c12)
except Exception:
    pass
layout["840"] = [1217, 1, 1314, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 45, 67)
    canvas.paste(_c13, (1155, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1155, 0, 1200, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 50, 60)
    canvas.paste(_c14, (1320, 3), _c14)
except Exception:
    pass
layout["icon_14"] = [1320, 3, 1370, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/15_icon_Wy.png
try:
    _c15 = get_crop(15, 51, 59)
    canvas.paste(_c15, (183, 3), _c15)
except Exception:
    pass
layout["Wy"] = [183, 3, 234, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/16_icon_New_York.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (288, 2792), _c16)
except Exception:
    pass
layout["New_York"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/17_icon_Nets_at_Knicks.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (864, 2792), _c17)
except Exception:
    pass
layout["Nets_at_Knicks"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/18_icon_iLliNoisE.png
try:
    _c18 = get_crop(18, 72, 72)
    canvas.paste(_c18, (906, 2406), _c18)
except Exception:
    pass
layout["iLliNoisE"] = [906, 2406, 978, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/19_icon_Tracking.png
try:
    _c19 = get_crop(19, 36, 72)
    canvas.paste(_c19, (1404, 2406), _c19)
except Exception:
    pass
layout["Tracking"] = [1404, 2406, 1440, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/20_icon_Recently_viewed_events.png
try:
    _c20 = get_crop(20, 72, 72)
    canvas.paste(_c20, (408, 2406), _c20)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 2406, 480, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/21_icon_S2_4_D.png
try:
    _c21 = get_crop(21, 114, 127)
    canvas.paste(_c21, (1139, 1731), _c21)
except Exception:
    pass
layout["S2_(#4_D="] = [1139, 1731, 1253, 1858]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/22_icon_TIcKETS.png
try:
    _c22 = get_crop(22, 1344, 840)
    canvas.paste(_c22, (48, 360), _c22)
except Exception:
    pass
layout["TIcKETS"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/23_icon_New_York.png
try:
    _c23 = get_crop(23, 462, 519)
    canvas.paste(_c23, (48, 2382), _c23)
except Exception:
    pass
layout["New_York"] = [48, 2382, 510, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/24_icon_Nets_at_Knicks.png
try:
    _c24 = get_crop(24, 288, 168)
    canvas.paste(_c24, (1152, 2792), _c24)
except Exception:
    pass
layout["Nets_at_Knicks"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/25_icon_New_York_NY.png
try:
    _c25 = get_crop(25, 391, 83)
    canvas.paste(_c25, (39, 121), _c25)
except Exception:
    pass
layout["New_York,_NY"] = [39, 121, 430, 204]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/26_icon_Madison_Square_Garden.png
try:
    _c26 = get_crop(26, 1309, 234)
    canvas.paste(_c26, (0, 1903), _c26)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 1903, 1309, 2137]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/27_text_7.40.png
try:
    _c27 = get_crop(27, 89, 41)
    canvas.paste(_c27, (22, 17), _c27)
except Exception:
    pass
layout["7.40"] = [22, 17, 111, 58]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/28_text_date.png
try:
    _c28 = get_crop(28, 114, 52)
    canvas.paste(_c28, (137, 208), _c28)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/29_text_Trending_events.png
try:
    _c29 = get_crop(29, 423, 79)
    canvas.paste(_c29, (38, 1303), _c29)
except Exception:
    pass
layout["Trending_events"] = [38, 1303, 461, 1382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/30_text_View_all.png
try:
    _c30 = get_crop(30, 264, 183)
    canvas.paste(_c30, (1176, 1248), _c30)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/31_text_Recently_viewed_events.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (408, 2406), _c31)
except Exception:
    pass
layout["Recently_viewed_events"] = [408, 2406, 480, 2478]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/32_text_View_all.png
try:
    _c32 = get_crop(32, 264, 183)
    canvas.paste(_c32, (1176, 2199), _c32)
except Exception:
    pass
layout["View_all"] = [1176, 2199, 1440, 2382]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/33_text_Illinoise.png
try:
    _c33 = get_crop(33, 170, 49)
    canvas.paste(_c33, (539, 2736), _c33)
except Exception:
    pass
layout["Illinoise"] = [539, 2736, 709, 2785]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/34_text_Nets_at_Knicks.png
try:
    _c34 = get_crop(34, 396, 519)
    canvas.paste(_c34, (1044, 2382), _c34)
except Exception:
    pass
layout["Nets_at_Knicks"] = [1044, 2382, 1440, 2901]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/35_clickable_Tickets.png
try:
    _c35 = get_crop(35, 288, 168)
    canvas.paste(_c35, (576, 2792), _c35)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/42f99a380cf348fea73bd58c0e4ec8db/step_01_2024_3_20_15_39_42f99a380cf348fea73bd58c0e4ec8db-4/36_clickable_View_all.png
try:
    _c36 = get_crop(36, 264, 7)
    canvas.paste(_c36, (1176, 2953), _c36)
except Exception:
    pass
layout["View_all"] = [1176, 2953, 1440, 2960]
