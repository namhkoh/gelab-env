# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_01
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4.png
# step_index: 1/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Top/baseline UI structure for the mobile SeatGeek-like screen
# Uses provided canvas (1440x2960) and draw

w, h = canvas.size

# Colors
bg_color = (251, 251, 251)         # overall app background (very light gray)
status_bar_color = (236, 239, 241) # status bar slightly darker
header_bg = (255, 255, 255)        # header white
divider = (233, 233, 233)          # subtle dividers
panel_bg = (255, 255, 255)         # white panels
muted_panel = (248, 248, 250)      # very slightly off-white panels
nav_bg = (255, 255, 255)           # bottom nav background
shadow = (240, 240, 240)           # subtle shadow strip

# Fill overall background
draw.rectangle([(0, 0), (w, h)], fill=bg_color)

# Status bar (top area where time/signal live)
status_h = 84
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_color)

# Header area (below status bar) with subtle bottom divider
header_top = status_h
header_h = 240
draw.rectangle([(0, header_top), (w, header_h)], fill=header_bg)
# bottom divider for header
draw.line([(24, header_h - 1), (w - 24, header_h - 1)], fill=divider, width=1)

# Main spacing/gutter under header left as background (do not draw hero card content)
# Add a faint horizontal guide line where the large hero card would sit (but not drawing card)
hero_start_y = 360  # hero card detected y; leave this area for pasted content

# "Just for you" section container background (subtle white rounded panel behind cards)
just_section_top = 1260
just_section_bottom = 1600
panel_margin = 24
draw.rounded_rectangle(
    [(panel_margin, just_section_top), (w - panel_margin, just_section_bottom)],
    radius=18, fill=panel_bg, outline=None
)
# Slight top shadow strip to separate from above content
draw.rectangle([(panel_margin + 6, just_section_top - 6), (w - panel_margin - 6, just_section_top)], fill=shadow)

# Divider line below the "Just for you" area (separates from trending header)
trend_header_y = just_section_bottom + 70
# A subtle horizontal divider across content width
draw.line([(24, trend_header_y - 18), (w - 24, trend_header_y - 18)], fill=divider, width=1)

# Trending events panel (large white panel with rounded corners)
trending_top = trend_header_y
trending_bottom = 2792  # top of bottom nav (detected)
draw.rounded_rectangle(
    [(12, trending_top), (w - 12, trending_bottom - 8)],
    radius=16, fill=panel_bg, outline=None
)

# Inner muted background behind the list items to give separation from page bg
draw.rectangle([(24, trending_top + 12), (w - 24, trending_bottom - 24)], fill=muted_panel)

# Horizontal separators between trending rows (match detected row heights)
# Detected rows at y=2197 (row1 height 236 -> separator at 2433), y=2433 (-> 2669), etc.
sep_x0 = 36
sep_x1 = w - 36
separators = [2433, 2669]  # y positions for separators (where rows end)
for y in separators:
    # draw a soft line across the content area
    draw.line([(sep_x0, y), (sep_x1, y)], fill=divider, width=1)

# Thin divider above trending list title area
draw.line([(24, trending_top + 8), (w - 24, trending_top + 8)], fill=divider, width=1)

# Bottom navigation bar background and top divider
nav_top = 2792
draw.rectangle([(0, nav_top), (w, h)], fill=nav_bg)
# top divider for nav
draw.line([(0, nav_top), (w, nav_top)], fill=divider, width=1)

# Subtle left/right content guides (margins) to match overall layout spacing
# These are faint and structural only
guide_color = (245, 245, 245)
draw.rectangle([(0, header_h), (12, h - 200)], fill=guide_color)
draw.rectangle([(w - 12, header_h), (w, h - 200)], fill=guide_color)

# Accent vertical separator near right for a thin panel hint (matches screenshot subtle edge)
draw.line([(w - 120, header_h + 20), (w - 120, trending_top)], fill=shadow, width=1)

# Provide subtle shadow band just above the bottom nav to lift content
draw.rectangle([(0, nav_top - 6), (w, nav_top)], fill=shadow)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/00_icon_Clippers.png
try:
    _c0 = get_crop(0, 1344, 840)
    canvas.paste(_c0, (48, 360), _c0)
except Exception:
    pass
layout["Clippers"] = [48, 360, 1392, 1200]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/01_icon_Dodger_Stadium.png
try:
    _c1 = get_crop(1, 1309, 236)
    canvas.paste(_c1, (0, 2197), _c1)
except Exception:
    pass
layout["Dodger_Stadium"] = [0, 2197, 1309, 2433]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/02_icon_View_all.png
try:
    _c2 = get_crop(2, 100, 151)
    canvas.paste(_c2, (1340, 2243), _c2)
except Exception:
    pass
layout["View_all"] = [1340, 2243, 1440, 2394]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/03_icon_Angel_Stadium_of_Anaheim.png
try:
    _c3 = get_crop(3, 1309, 236)
    canvas.paste(_c3, (0, 2433), _c3)
except Exception:
    pass
layout["Angel_Stadium_of_Anaheim"] = [0, 2433, 1309, 2669]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/04_icon_Los_Angeles_CA.png
try:
    _c4 = get_crop(4, 61, 58)
    canvas.paste(_c4, (243, 5), _c4)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [243, 5, 304, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/05_icon_S262.png
try:
    _c5 = get_crop(5, 462, 519)
    canvas.paste(_c5, (48, 1431), _c5)
except Exception:
    pass
layout["S262+"] = [48, 1431, 510, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/06_icon_7.48_W.png
try:
    _c6 = get_crop(6, 54, 56)
    canvas.paste(_c6, (115, 6), _c6)
except Exception:
    pass
layout["7.48_W"] = [115, 6, 169, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/07_icon_888.png
try:
    _c7 = get_crop(7, 97, 63)
    canvas.paste(_c7, (1216, 1), _c7)
except Exception:
    pass
layout["888"] = [1216, 1, 1313, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/08_icon_7.48_W.png
try:
    _c8 = get_crop(8, 47, 56)
    canvas.paste(_c8, (185, 6), _c8)
except Exception:
    pass
layout["7.48_W"] = [185, 6, 232, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 50, 62)
    canvas.paste(_c9, (1320, 3), _c9)
except Exception:
    pass
layout["icon_9"] = [1320, 3, 1370, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/10_icon_888.png
try:
    _c10 = get_crop(10, 144, 240)
    canvas.paste(_c10, (1260, 72), _c10)
except Exception:
    pass
layout["888"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/11_icon_Tracking.png
try:
    _c11 = get_crop(11, 288, 168)
    canvas.paste(_c11, (864, 2792), _c11)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 103, 150)
    canvas.paste(_c12, (1337, 2480), _c12)
except Exception:
    pass
layout["icon_12"] = [1337, 2480, 1440, 2630]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 54, 58)
    canvas.paste(_c13, (314, 5), _c13)
except Exception:
    pass
layout["icon_13"] = [314, 5, 368, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/14_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (288, 2792), _c14)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/15_icon_7_PM.png
try:
    _c15 = get_crop(15, 264, 183)
    canvas.paste(_c15, (1176, 2014), _c15)
except Exception:
    pass
layout["7_PM"] = [1176, 2014, 1440, 2197]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 45, 64)
    canvas.paste(_c16, (1155, 1), _c16)
except Exception:
    pass
layout["icon_16"] = [1155, 1, 1200, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/17_icon_S66.png
try:
    _c17 = get_crop(17, 462, 533)
    canvas.paste(_c17, (546, 1431), _c17)
except Exception:
    pass
layout["S66+"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/18_icon_Browse.png
try:
    _c18 = get_crop(18, 288, 162)
    canvas.paste(_c18, (0, 2792), _c18)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/19_icon_W_Conf_Ist_Rnd.png
try:
    _c19 = get_crop(19, 462, 533)
    canvas.paste(_c19, (546, 1431), _c19)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:"] = [546, 1431, 1008, 1964]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 100, 118)
    canvas.paste(_c20, (1340, 2707), _c20)
except Exception:
    pass
layout["icon_20"] = [1340, 2707, 1440, 2825]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/21_icon_icon_21.png
try:
    _c21 = get_crop(21, 116, 127)
    canvas.paste(_c21, (1138, 2495), _c21)
except Exception:
    pass
layout["icon_21"] = [1138, 2495, 1254, 2622]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/22_icon_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c22 = get_crop(22, 288, 168)
    canvas.paste(_c22, (576, 2792), _c22)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/23_icon_Account.png
try:
    _c23 = get_crop(23, 288, 168)
    canvas.paste(_c23, (1152, 2792), _c23)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/24_text_Los_Angeles_CA.png
try:
    _c24 = get_crop(24, 456, 80)
    canvas.paste(_c24, (44, 132), _c24)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [44, 132, 500, 212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/25_text_Just_for_you.png
try:
    _c25 = get_crop(25, 309, 66)
    canvas.paste(_c25, (38, 1310), _c25)
except Exception:
    pass
layout["Just_for_you"] = [38, 1310, 347, 1376]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/26_text_View_all.png
try:
    _c26 = get_crop(26, 264, 183)
    canvas.paste(_c26, (1176, 1248), _c26)
except Exception:
    pass
layout["View_all"] = [1176, 1248, 1440, 1431]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/27_text_W_Conf_Ist_Rnd_Mavericks_at_Clippers.png
try:
    _c27 = get_crop(27, 288, 168)
    canvas.paste(_c27, (576, 2792), _c27)
except Exception:
    pass
layout["W_Conf_Ist_Rnd:_Mavericks"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/28_clickable_Tracking.png
try:
    _c28 = get_crop(28, 396, 519)
    canvas.paste(_c28, (1044, 1431), _c28)
except Exception:
    pass
layout["Tracking"] = [1044, 1431, 1440, 1950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/29_clickable_Tracking.png
try:
    _c29 = get_crop(29, 72, 72)
    canvas.paste(_c29, (408, 1455), _c29)
except Exception:
    pass
layout["Tracking"] = [408, 1455, 480, 1527]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_01_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-4/30_clickable_Tracking.png
try:
    _c30 = get_crop(30, 72, 72)
    canvas.paste(_c30, (906, 1455), _c30)
except Exception:
    pass
layout["Tracking"] = [906, 1455, 978, 1527]
