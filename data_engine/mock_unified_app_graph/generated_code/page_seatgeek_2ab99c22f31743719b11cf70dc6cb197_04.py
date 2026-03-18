# page_id: page_seatgeek_2ab99c22f31743719b11cf70dc6cb197_04
# screenshot: 2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7.png
# step_index: 4/6
# task: Open SeatGeek. Search "Oracle Arena". Add the venue to the watch list.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background, status bar, headers, section cards, separators, and bottom nav background
w, h = canvas.size

# Colors
bg_color = "#ffffff"            # main background (white)
status_bar_color = "#efefef"    # top status area
header_bg = "#fbfbfb"           # header/toolbar background
divider_color = "#e6e6e6"       # subtle dividers
card_bg = "#ffffff"             # cards are white on white canvas
card_shadow = (230, 230, 230)   # very light shadow color for card edges
bottom_nav_bg = "#ffffff"       # bottom navigation background
muted_bg = "#fafafa"            # slightly off-white sections

# Fill entire canvas (canvas is already white, but ensure color)
draw.rectangle((0, 0, w, h), fill=bg_color)

# Status bar area at top (~84px)
status_h = 84
draw.rectangle((0, 0, w, status_h), fill=status_bar_color)

# Header / toolbar background (area behind search box)
header_top = status_h
header_bottom = 270
draw.rectangle((0, header_top, w, header_bottom), fill=header_bg)

# Thin divider under header
draw.line((32, header_bottom, w - 32, header_bottom), fill=divider_color, width=2)

# Helper for rounded card drawing with subtle top/bottom hairline to emulate card boundaries
def rounded_card(x0, y0, x1, y1, radius=18, fill=card_bg):
    # main rounded rectangle
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=fill)
    # subtle top and bottom hairlines for separation
    draw.line((x0 + 16, y0 + 1, x1 - 16, y0 + 1), fill=divider_color, width=1)
    draw.line((x0 + 16, y1 - 1, x1 - 16, y1 - 1), fill=divider_color, width=1)

# Top results card (group behind list items)
card_margin_x = 32
top_card_y0 = 320
top_card_y1 = 860
rounded_card(card_margin_x, top_card_y0, w - card_margin_x, top_card_y1, radius=16, fill=card_bg)

# Draw separators between entries in Top results card
# There are typically three list items in this card; draw two separators
sep_x0 = card_margin_x + 112  # leave room where avatar would be (background only)
sep_x1 = w - card_margin_x - 32
sep_y_positions = [top_card_y0 + 140, top_card_y0 + 280]
for y_pos in sep_y_positions:
    draw.line((sep_x0, y_pos, sep_x1, y_pos), fill=divider_color, width=1)

# Subtle dividing line below Top results section (full width)
divider_y = top_card_y1 + 24
draw.line((32, divider_y, w - 32, divider_y), fill=divider_color, width=1)

# Performers section card
perf_card_y0 = divider_y + 28
perf_card_y1 = perf_card_y0 + 200
rounded_card(card_margin_x, perf_card_y0, w - card_margin_x, perf_card_y1, radius=16, fill=card_bg)

# Separator under performers
draw.line((32, perf_card_y1 + 24, w - 32, perf_card_y1 + 24), fill=divider_color, width=1)

# Events card (single event style card)
events_card_y0 = perf_card_y1 + 56
events_card_y1 = events_card_y0 + 160
# For the event item, emulate the darker thumbnail area on the left as background only
# Thumbnail background (dark rounded rect)
thumb_x0 = card_margin_x + 16
thumb_x1 = thumb_x0 + 120
thumb_y0 = events_card_y0 + 16
thumb_y1 = events_card_y1 - 16
draw.rounded_rectangle((thumb_x0, thumb_y0, thumb_x1, thumb_y1), radius=14, fill="#0b0b0b")

# Light card background on the rest of the event row
draw.rounded_rectangle((card_margin_x, events_card_y0, w - card_margin_x, events_card_y1), radius=12, fill=card_bg)
# Separator below events
draw.line((32, events_card_y1 + 24, w - 32, events_card_y1 + 24), fill=divider_color, width=1)

# Venues card (list with three items)
venues_y0 = events_card_y1 + 56
venues_y1 = venues_y0 + 420
rounded_card(card_margin_x, venues_y0, w - card_margin_x, venues_y1, radius=16, fill=card_bg)
# separators between venue rows (three items -> two separators)
venue_sep_ys = [venues_y0 + 130, venues_y0 + 260]
for y in venue_sep_ys:
    draw.line((sep_x0, y, sep_x1, y), fill=divider_color, width=1)

# Separator under venues
draw.line((32, venues_y1 + 24, w - 32, venues_y1 + 24), fill=divider_color, width=1)

# Recent searches header area (just spacing + subtle bg)
recent_y0 = venues_y1 + 48
recent_y1 = recent_y0 + 300
# Slight muted background band for recent searches area
draw.rectangle((0, recent_y0, w, recent_y1), fill=muted_bg)
# small divider under recent searches band
draw.line((32, recent_y1, w - 32, recent_y1), fill=divider_color, width=1)

# Bottom navigation bar background (space for icons will be pasted on top)
bottom_nav_top = 2792 - 40  # a bit above the icons area to create a top border/shadow
draw.rectangle((0, bottom_nav_top, w, h), fill=bottom_nav_bg)
# top hairline for nav bar
draw.line((0, bottom_nav_top + 1, w, bottom_nav_top + 1), fill=divider_color, width=2)

# small shadow above major content area (under header)
shadow_y = header_bottom + 6
draw.line((32, shadow_y, w - 32, shadow_y), fill=divider_color, width=1)

# End of drawing operations

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/00_icon_Top_results.png
try:
    _c0 = get_crop(0, 1440, 179)
    canvas.paste(_c0, (0, 471), _c0)
except Exception:
    pass
layout["Top_results"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/01_icon_Performers.png
try:
    _c1 = get_crop(1, 1440, 179)
    canvas.paste(_c1, (0, 1217), _c1)
except Exception:
    pass
layout["Performers"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/02_icon_Oracle_Arenal.png
try:
    _c2 = get_crop(2, 1032, 144)
    canvas.paste(_c2, (216, 120), _c2)
except Exception:
    pass
layout["Oracle_Arenal"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 60, 62)
    canvas.paste(_c3, (244, 3), _c3)
except Exception:
    pass
layout["icon_3"] = [244, 3, 304, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 43, 70)
    canvas.paste(_c4, (1155, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1155, 0, 1198, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/05_icon_Fri.png
try:
    _c5 = get_crop(5, 1440, 179)
    canvas.paste(_c5, (0, 1605), _c5)
except Exception:
    pass
layout["Fri,"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/06_icon_8.30_my.png
try:
    _c6 = get_crop(6, 168, 144)
    canvas.paste(_c6, (48, 120), _c6)
except Exception:
    pass
layout["8.30_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/07_icon_Recent_searches.png
try:
    _c7 = get_crop(7, 288, 162)
    canvas.paste(_c7, (288, 2792), _c7)
except Exception:
    pass
layout["Recent_searches"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 93, 69)
    canvas.paste(_c8, (1219, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1219, 0, 1312, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 54, 61)
    canvas.paste(_c9, (315, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [315, 3, 369, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/10_icon_Oracle_Arena_O.co_Coliseum.png
try:
    _c10 = get_crop(10, 1440, 179)
    canvas.paste(_c10, (0, 471), _c10)
except Exception:
    pass
layout["Oracle_Arena_&_O.co_Colis"] = [0, 471, 1440, 650]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/11_icon_8.30_my.png
try:
    _c11 = get_crop(11, 46, 61)
    canvas.paste(_c11, (186, 2), _c11)
except Exception:
    pass
layout["8.30_my"] = [186, 2, 232, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/12_icon_Clear.png
try:
    _c12 = get_crop(12, 144, 144)
    canvas.paste(_c12, (1248, 120), _c12)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/13_icon_Recent_searches.png
try:
    _c13 = get_crop(13, 288, 168)
    canvas.paste(_c13, (0, 2792), _c13)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/14_icon_Sutton_WV.png
try:
    _c14 = get_crop(14, 1440, 179)
    canvas.paste(_c14, (0, 1605), _c14)
except Exception:
    pass
layout["Sutton,_WV"] = [0, 1605, 1440, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/15_icon_Oracle_Arena_O.co_Coliseum.png
try:
    _c15 = get_crop(15, 1440, 179)
    canvas.paste(_c15, (0, 1993), _c15)
except Exception:
    pass
layout["Oracle_Arena_&_O.co_Colis"] = [0, 1993, 1440, 2172]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 65)
    canvas.paste(_c16, (1326, 3), _c16)
except Exception:
    pass
layout["icon_16"] = [1326, 3, 1371, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/17_icon_Account.png
try:
    _c17 = get_crop(17, 288, 168)
    canvas.paste(_c17, (1152, 2792), _c17)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/18_icon_Oracle_Arena_O.co_Coliseum.png
try:
    _c18 = get_crop(18, 1440, 179)
    canvas.paste(_c18, (0, 650), _c18)
except Exception:
    pass
layout["Oracle_Arena_&_O.co_Colis"] = [0, 650, 1440, 829]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/19_icon_Bay.png
try:
    _c19 = get_crop(19, 1440, 179)
    canvas.paste(_c19, (0, 2172), _c19)
except Exception:
    pass
layout["Bay"] = [0, 2172, 1440, 2351]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/20_icon_Oracle_Arena.png
try:
    _c20 = get_crop(20, 1440, 179)
    canvas.paste(_c20, (0, 1217), _c20)
except Exception:
    pass
layout["Oracle_Arena"] = [0, 1217, 1440, 1396]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/21_icon_Tracking.png
try:
    _c21 = get_crop(21, 288, 168)
    canvas.paste(_c21, (864, 2792), _c21)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/22_icon_8.30_my.png
try:
    _c22 = get_crop(22, 52, 62)
    canvas.paste(_c22, (117, 1), _c22)
except Exception:
    pass
layout["8.30_my"] = [117, 1, 169, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/23_icon_Tickets.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (576, 2792), _c23)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/24_text_Top_results.png
try:
    _c24 = get_crop(24, 295, 72)
    canvas.paste(_c24, (40, 373), _c24)
except Exception:
    pass
layout["Top_results"] = [40, 373, 335, 445]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/25_text_Oracle_Arena.png
try:
    _c25 = get_crop(25, 293, 53)
    canvas.paste(_c25, (236, 865), _c25)
except Exception:
    pass
layout["Oracle_Arena"] = [236, 865, 529, 918]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/26_text_Area_Latino_Music_Festival.png
try:
    _c26 = get_crop(26, 1440, 179)
    canvas.paste(_c26, (0, 829), _c26)
except Exception:
    pass
layout["Area_Latino_Music_Festiva"] = [0, 829, 1440, 1008]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/27_text_No_events.png
try:
    _c27 = get_crop(27, 201, 43)
    canvas.paste(_c27, (239, 931), _c27)
except Exception:
    pass
layout["No_events"] = [239, 931, 440, 974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/28_text_Performers.png
try:
    _c28 = get_crop(28, 293, 54)
    canvas.paste(_c28, (44, 1122), _c28)
except Exception:
    pass
layout["Performers"] = [44, 1122, 337, 1176]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/29_text_Events.png
try:
    _c29 = get_crop(29, 177, 54)
    canvas.paste(_c29, (46, 1510), _c29)
except Exception:
    pass
layout["Events"] = [46, 1510, 223, 1564]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/30_text_Venues.png
try:
    _c30 = get_crop(30, 195, 62)
    canvas.paste(_c30, (44, 1897), _c30)
except Exception:
    pass
layout["Venues"] = [44, 1897, 239, 1959]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/31_text_Arlene_and_Robert_Kogod_Cradle_at_Arena_.png
try:
    _c31 = get_crop(31, 1440, 179)
    canvas.paste(_c31, (0, 2351), _c31)
except Exception:
    pass
layout["Arlene_and_Robert_Kogod_C"] = [0, 2351, 1440, 2530]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/32_text_No_events.png
try:
    _c32 = get_crop(32, 201, 43)
    canvas.paste(_c32, (239, 2452), _c32)
except Exception:
    pass
layout["No_events"] = [239, 2452, 440, 2495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/33_text_Recent_searches.png
try:
    _c33 = get_crop(33, 288, 168)
    canvas.paste(_c33, (0, 2792), _c33)
except Exception:
    pass
layout["Recent_searches"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/2ab99c22f31743719b11cf70dc6cb197/step_04_2024_4_22_20_29_2ab99c22f31743719b11cf70dc6cb197-7/34_text_Bay.png
try:
    _c34 = get_crop(34, 98, 61)
    canvas.paste(_c34, (557, 863), _c34)
except Exception:
    pass
layout["Bay"] = [557, 863, 655, 924]
