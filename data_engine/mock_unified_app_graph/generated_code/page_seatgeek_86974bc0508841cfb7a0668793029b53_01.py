# page_id: page_seatgeek_86974bc0508841cfb7a0668793029b53_01
# screenshot: 2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4.png
# step_index: 1/5
# task: Open SeatGeek. Search for the "Ed Sheeran" concert. Check the next upcoming event. When and where is the concert?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas.
# Assumes: canvas (PIL Image RGB 1440x2960) and draw (ImageDraw) are provided.

# Colors
BG = (250, 250, 250)        # overall page background (very light)
STATUS_BG = (236, 236, 236) # status bar background (light grey)
HEADER_BG = (255, 255, 255) # header/toolbar background (white)
DIVIDER = (228, 228, 228)   # thin divider lines
CARD_BLUE = (47, 90, 168)   # hero card / banner (deep blue)
CARD_SHADOW = (220, 225, 235) # subtle shadow color for large cards
SECTION_BG = (255, 255, 255) # section card backgrounds (white)
LIST_BG = (255, 255, 255)    # trending list background
BOTTOM_BAR_BG = (255, 255, 255)
SUBTLE = (245, 245, 245)

w, h = canvas.size

# Fill overall background
draw.rectangle([0, 0, w, h], fill=BG)

# Status bar area (top ~80px)
status_h = 80
draw.rectangle([0, 0, w, status_h], fill=STATUS_BG)

# Header / toolbar area below status bar (~120px tall)
header_top = status_h
header_bottom = 220
draw.rectangle([0, header_top, w, header_bottom], fill=HEADER_BG)
# bottom divider for header
draw.line([(24, header_bottom), (w-24, header_bottom)], fill=DIVIDER, width=1)

# Large hero/banner card background (slightly offset to avoid exact duplication of pasted asset)
# Detected hero icon is at (48,360) size 1344x840; draw an offset rounded card behind it.
hero_x = 36
hero_y = 320
hero_w = 1368
hero_h = 880
hero_radius = 28
# subtle outer shadow rectangle behind hero
draw.rounded_rectangle(
    [hero_x+6, hero_y+10, hero_x+hero_w+6, hero_y+hero_h+10],
    radius=hero_radius+4,
    fill=CARD_SHADOW
)
# main hero background (a colored banner)
draw.rounded_rectangle(
    [hero_x, hero_y, hero_x+hero_w, hero_y+hero_h],
    radius=hero_radius,
    fill=CARD_BLUE
)

# Section: "Just for you" container background (white card area)
just_top = 1320
just_left = 24
just_right = w - 24
just_height = 420
draw.rounded_rectangle(
    [just_left, just_top, just_right, just_top + just_height],
    radius=14,
    fill=SECTION_BG
)
# subtle top divider for the section
draw.line([(just_left+20, just_top+72), (just_right-20, just_top+72)], fill=SUBTLE, width=1)

# Thumbnails row placeholders for "Just for you" (three light card backgrounds, slightly offset from detected image crops)
thumb_w = 460
thumb_h = 320
thumb_radius = 20
thumb_gap = 36
thumb_top = just_top + 110
thumb_x1 = 36
thumb_x2 = thumb_x1 + thumb_w + thumb_gap
thumb_x3 = thumb_x2 + thumb_w + thumb_gap

# Left placeholder (slightly different size/pos than detected to avoid duplication)
draw.rounded_rectangle([thumb_x1-8, thumb_top, thumb_x1-8 + thumb_w + 16, thumb_top + thumb_h],
                       radius=thumb_radius, fill=(255,255,255))
# Middle placeholder
draw.rounded_rectangle([thumb_x2-8, thumb_top, thumb_x2-8 + thumb_w + 16, thumb_top + thumb_h],
                       radius=thumb_radius, fill=(255,255,255))
# Right placeholder
draw.rounded_rectangle([thumb_x3-8, thumb_top, thumb_x3-8 + thumb_w + 16, thumb_top + thumb_h],
                       radius=thumb_radius, fill=(255,255,255))

# Divider line under thumbnails / section
draw.line([(24, just_top + just_height - 2), (w-24, just_top + just_height - 2)], fill=DIVIDER, width=1)

# Trending events list container (white background, rounded)
list_top = 2040
list_left = 24
list_right = w - 24
list_height = 740
list_radius = 12
draw.rounded_rectangle([list_left, list_top, list_right, list_top + list_height],
                       radius=list_radius, fill=LIST_BG)

# Section title divider (light)
draw.line([(list_left+20, list_top+80), (list_right-20, list_top+80)], fill=SUBTLE, width=1)

# Rows separators inside trending list (positions chosen near where detected rows will be pasted)
row_start_y = list_top + 120
row_height = 236
for i in range(0, 4):
    y = row_start_y + i * row_height
    # Draw light separator line
    draw.line([(list_left+24, y), (list_right-24, y)], fill=DIVIDER, width=1)

# Right-hand "View all" area spacer (light divider)
view_all_x = list_right - 160
draw.line([(view_all_x, list_top+20), (view_all_x, list_top + list_height - 20)], fill=SUBTLE, width=1)

# Bottom navigation bar background (top border line + white background)
bottom_top = 2792
draw.line([(0, bottom_top), (w, bottom_top)], fill=DIVIDER, width=1)
draw.rectangle([0, bottom_top, w, h], fill=BOTTOM_BAR_BG)

# Slight highlights on edges to separate sections visually
draw.line([(24, hero_y - 24), (w-24, hero_y - 24)], fill=SUBTLE, width=1)
draw.line([(24, just_top - 12), (w-24, just_top - 12)], fill=SUBTLE, width=1)

# Finished drawing structural elements.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/00_icon_Clippers.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Clippers"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/01_icon_Dodger_Stadium.png
try:
    _c1 = get_crop(1, 1309, 236)
    canvas.paste(_c1, (0, 2197), _c1)
except Exception:
    pass
layout["Dodger_Stadium"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 100, 151)
    canvas.paste(_c2, (1340, 2243), _c2)
except Exception:
    pass
layout["View_all"] = [1340, 2243, 1440, 2394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/03_icon_Angel_Stadium_of_Anaheim.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2433), _c3)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/04_icon_Los_Angeles_CA.png
try:
    _c4 = get_crop(4, 61, 58)
    canvas.paste(_c4, (243, 5), _c4)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/05_icon_S262.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (48, 1431), _c5)
except Exception:
    pass
layout["S262+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/06_icon_8.00_my.png
try:
    _c6 = get_crop(6, 53, 56)
    canvas.paste(_c6, (116, 6), _c6)
except Exception:
    pass
layout["8.00_my"] = [116, 6, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/07_icon_888.png
try:
    _c7 = get_crop(7, 98, 63)
    canvas.paste(_c7, (1216, 1), _c7)
except Exception:
    pass
layout["888"] = [1216, 1, 1314, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/08_icon_8.00_my.png
try:
    _c8 = get_crop(8, 46, 56)
    canvas.paste(_c8, (186, 6), _c8)
except Exception:
    pass
layout["8.00_my"] = [186, 6, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 62)
    canvas.paste(_c9, (1320, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1320, 3, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/10_icon_888.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/11_icon_Tracking.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (864, 2792), _c11)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 54, 57)
    canvas.paste(_c12, (314, 6), _c12)
except Exception:
    pass
layout["icon_12"] = [314, 6, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 103, 150)
    canvas.paste(_c13, (1337, 2480), _c13)
except Exception:
    pass
layout["icon_13"] = [1337, 2480, 1440, 2630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/14_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (288, 2792), _c14)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/15_icon_7_PM.png
try:
    _c15 = get_crop(15, 264, 183)
    canvas.paste(_c15, (1176, 2014), _c15)
except Exception:
    pass
layout["7_PM"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 64)
    canvas.paste(_c16, (1155, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1155, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/17_icon_S66.png
try:
    _c17 = get_crop(17, 462, 533)
    canvas.paste(_c17, (546, 1431), _c17)
except Exception:
    pass
layout["S66+"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/18_icon_Browse.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/19_icon_W_Conf_Ist_Rnd.png
try:
    _c19 = get_crop(19, 462, 533)
    canvas.paste(_c19, (546, 1431), _c19)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 116, 127)
    canvas.paste(_c20, (1138, 2495), _c20)
except Exception:
    pass
layout["icon_20"] = [1138, 2495, 1254, 2622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 100, 118)
    canvas.paste(_c21, (1340, 2707), _c21)
except Exception:
    pass
layout["icon_21"] = [1340, 2707, 1440, 2825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/22_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (576, 2792), _c22)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/23_icon_Account.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (1152, 2792), _c23)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/24_icon_Los_Angeles_CA.png
try:
    _c24 = get_crop(24, 461, 84)
    canvas.paste(_c24, (41, 122), _c24)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [41, 122, 502, 206]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/25_text_date.png
try:
    _c25 = get_crop(25, 114, 52)
    canvas.paste(_c25, (137, 208), _c25)
except Exception:
    pass
layout["date"] = [137, 208, 251, 260]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/26_text_Just_for_you.png
try:
    _c26 = get_crop(26, 309, 66)
    canvas.paste(_c26, (38, 1310), _c26)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 347, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/27_text_View_all.png
try:
    _c27 = get_crop(27, 264, 183)
    canvas.paste(_c27, (1176, 1248), _c27)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/28_text_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c28 = get_crop(28, 288, 168)
    canvas.paste(_c28, (576, 2792), _c28)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 396, 519)
    canvas.paste(_c29, (1044, 1431), _c29)
except Exception:
    pass
layout["Tracking"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (408, 1455), _c30)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/86974bc0508841cfb7a0668793029b53/step_01_2024_4_22_19_59_86974bc0508841cfb7a0668793029b53-4/31_clickable_Tracking.png
try:
    _c31 = get_crop(31, 72, 72)
    canvas.paste(_c31, (906, 1455), _c31)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
