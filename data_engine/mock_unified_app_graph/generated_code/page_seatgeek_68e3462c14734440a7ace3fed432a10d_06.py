# page_id: page_seatgeek_68e3462c14734440a7ace3fed432a10d_06
# screenshot: 2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9.png
# step_index: 6/13
# task: Open SeatGeek and change the current location to Los Angeles. Then find the first concert show and track its performer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Canvas already provided: 1440x2960 RGB, white background
# draw is the ImageDraw object. font_sm, font_md, font_lg, font_xl available (not used).

# Colors
WHITE = (255, 255, 255)
OFF_WHITE = (250, 250, 250)
LIGHT_GREY = (238, 238, 238)
MID_GREY = (230, 230, 230)
PANEL_GREY = (245, 245, 246)
DIVIDER = (225, 225, 225)
DARK_STRIP = (28, 28, 30)
CARD_DARK = (20, 20, 22)

# Fill overall background (ensure clean)
draw.rectangle([(0, 0), (1440, 2960)], fill=WHITE)

# Status bar area (top ~56px) - subtle grey
status_h = 56
draw.rectangle([(0, 0), (1440, status_h)], fill=LIGHT_GREY)

# Header / toolbar area below status (white with subtle bottom divider/shadow)
header_top = status_h
header_bottom = 160
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=WHITE)
# thin divider line
draw.line([(0, header_bottom), (1440, header_bottom)], fill=DIVIDER, width=1)
# slight shadow under divider
draw.line([(0, header_bottom+1), (1440, header_bottom+1)], fill=(245,245,245), width=1)

# Location input card (rounded) - behind the "Location" input area
loc_x0, loc_y0 = 40, 200
loc_x1, loc_y1 = 1400, 284
draw.rounded_rectangle([(loc_x0, loc_y0), (loc_x1, loc_y1)],
                       radius=14, fill=WHITE, outline=MID_GREY)

# Subtle divider under location input
draw.line([(loc_x0+8, loc_y1+18), (loc_x1-8, loc_y1+18)], fill=DIVIDER, width=1)

# Date selection card (rounded container for Today/Tomorrow/Weekend) --
# keep only background and separators (no text)
date_x0, date_y0 = 40, 480
date_x1, date_y1 = 1400, 640
draw.rounded_rectangle([(date_x0, date_y0), (date_x1, date_y1)],
                       radius=16, fill=WHITE, outline=MID_GREY)
# central thin separator line (for visual separation inside the card)
mid_line_y = date_y0 + (date_y1 - date_y0) // 2 + 10
draw.line([(date_x0+24, mid_line_y), (date_x1-24, mid_line_y)], fill=DIVIDER, width=1)

# Dark category strip background (behind the three category cards)
cat_strip_top = 880
cat_strip_bottom = 1220
draw.rectangle([(0, cat_strip_top), (1440, cat_strip_bottom)], fill=OFF_WHITE)
# place a darker panel centered vertically for the cards
panel_inset = 28
panel_top = cat_strip_top + 36
panel_bottom = cat_strip_bottom - 36
draw.rectangle([(0, cat_strip_top), (1440, panel_top)], fill=OFF_WHITE)  # ensure clean above
# dark rounded background band to contrast cards (subtle)
band_x0, band_x1 = 24, 1440-24
band_y0, band_y1 = panel_top - 6, panel_bottom + 6
draw.rectangle([(band_x0, band_y0), (band_x1, band_y1)], fill=(245,245,245))

# Draw three dark rounded card backgrounds for categories (only backgrounds)
card_w = 420
card_h = 260
gap = 24
left_margin = 42
card_radius = 22
# positions for three cards
x = left_margin
y = panel_top
for i in range(3):
    draw.rounded_rectangle([(x, y), (x + card_w, y + card_h)],
                           radius=card_radius, fill=CARD_DARK, outline=(40,40,40))
    # faint inner overlay to suggest image framing (no content)
    inner_pad = 12
    draw.rounded_rectangle([(x+inner_pad, y+inner_pad), (x+card_w-inner_pad, y+card_h-inner_pad)],
                           radius=card_radius-6, outline=(60,60,60))
    x += card_w + gap

# Separator line below category area
sep_y = cat_strip_bottom + 6
draw.line([(28, sep_y), (1440-28, sep_y)], fill=DIVIDER, width=1)

# "Just announced" section background area (light panel)
just_top = 1560
just_bottom = 2120
draw.rectangle([(0, just_top), (1440, just_bottom)], fill=WHITE)
# Subtle area behind the list card (a pale panel)
panel_pad = 34
draw.rectangle([(panel_pad, just_top+40), (1440-panel_pad, just_top+240)], fill=PANEL_GREY, outline=(235,235,235), )
# small rounded thumbnail placeholder background for the item (no image content)
thumb_x0 = panel_pad + 16
thumb_y0 = just_top + 56
thumb_x1 = thumb_x0 + 220
thumb_y1 = thumb_y0 + 150
draw.rounded_rectangle([(thumb_x0, thumb_y0), (thumb_x1, thumb_y1)], radius=14, fill=(230,230,235))

# Thin divider under the "Just announced" list
divider_y = just_top + 260
draw.line([(24, divider_y), (1440-24, divider_y)], fill=DIVIDER, width=1)

# "Sports" section area background (light)
sports_top = divider_y + 24
sports_bottom = sports_top + 640
draw.rectangle([(0, sports_top), (1440, sports_bottom)], fill=WHITE)
# a row of card placeholders for sports thumbnails (3 across) as rounded rect placeholders
card_w2 = 420
card_h2 = 220
gap2 = 28
x2 = 36
y2 = sports_top + 36
for i in range(3):
    draw.rounded_rectangle([(x2, y2), (x2 + card_w2, y2 + card_h2)],
                           radius=18, fill=(245,245,247), outline=(230,230,230))
    x2 += card_w2 + gap2

# Final thin divider above bottom navigation
nav_top = 2792
draw.line([(0, nav_top), (1440, nav_top)], fill=DIVIDER, width=1)
# Bottom navigation bar background (white) with slight shadow line above
draw.rectangle([(0, nav_top), (1440, 2960)], fill=WHITE)
draw.line([(0, nav_top+2), (1440, nav_top+2)], fill=(248,248,248), width=1)

# A subtle elevated background for the center of the nav (no icons)
nav_center_bg_w = 68
nav_center_bg_h = 68
nav_center_x = 720 - nav_center_bg_w//2
nav_center_y = nav_top + 18
draw.ellipse([(nav_center_x, nav_center_y), (nav_center_x+nav_center_bg_w, nav_center_y+nav_center_bg_h)],
             fill=(255,255,255), outline=(235,235,235))

# Light overlays and shadows to make sections feel separated (soft)
# soft top shadow under header
draw.rectangle([(0, header_bottom), (1440, header_bottom+4)], fill=(248,248,248))

# End of background and structure drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/00_icon_Broadway.png
try:
    _c0 = get_crop(0, 404, 318)
    canvas.paste(_c0, (1036, 1261), _c0)
except Exception:
    pass
layout["Broadway"] = [1036, 1261, 1440, 1579]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/01_icon_Concerts.png
try:
    _c1 = get_crop(1, 492, 149)
    canvas.paste(_c1, (474, 1052), _c1)
except Exception:
    pass
layout["Concerts"] = [474, 1052, 966, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/02_icon_Sports.png
try:
    _c2 = get_crop(2, 471, 321)
    canvas.paste(_c2, (42, 1260), _c2)
except Exception:
    pass
layout["Sports"] = [42, 1260, 513, 1581]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/03_icon_Tomorrow.png
try:
    _c3 = get_crop(3, 1344, 153)
    canvas.paste(_c3, (48, 505), _c3)
except Exception:
    pass
layout["Tomorrow"] = [48, 505, 1392, 658]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 49, 68)
    canvas.paste(_c4, (1153, 0), _c4)
except Exception:
    pass
layout["icon_4"] = [1153, 0, 1202, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/05_icon_8.30_my.png
try:
    _c5 = get_crop(5, 57, 58)
    canvas.paste(_c5, (181, 4), _c5)
except Exception:
    pass
layout["8.30_my"] = [181, 4, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/06_icon_8.30_my.png
try:
    _c6 = get_crop(6, 52, 59)
    canvas.paste(_c6, (116, 3), _c6)
except Exception:
    pass
layout["8.30_my"] = [116, 3, 168, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 52, 61)
    canvas.paste(_c7, (1320, 3), _c7)
except Exception:
    pass
layout["icon_7"] = [1320, 3, 1372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 103, 67)
    canvas.paste(_c8, (1212, 0), _c8)
except Exception:
    pass
layout["icon_8"] = [1212, 0, 1315, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/09_icon_55.png
try:
    _c9 = get_crop(9, 288, 168)
    canvas.paste(_c9, (576, 2792), _c9)
except Exception:
    pass
layout["55"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/10_icon_Tracking.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (864, 2792), _c10)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/11_icon_Browse.png
try:
    _c11 = get_crop(11, 288, 162)
    canvas.paste(_c11, (0, 2792), _c11)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/12_icon_55.png
try:
    _c12 = get_crop(12, 288, 168)
    canvas.paste(_c12, (288, 2792), _c12)
except Exception:
    pass
layout["55"] = [288, 2792, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/13_icon_Close.png
try:
    _c13 = get_crop(13, 144, 240)
    canvas.paste(_c13, (1260, 72), _c13)
except Exception:
    pass
layout["Close"] = [1260, 72, 1404, 312]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/15_icon_Los_Angeles_CA.png
try:
    _c15 = get_crop(15, 52, 58)
    canvas.paste(_c15, (246, 5), _c15)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [246, 5, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/16_icon_Today.png
try:
    _c16 = get_crop(16, 448, 149)
    canvas.paste(_c16, (48, 901), _c16)
except Exception:
    pass
layout["Today"] = [48, 901, 496, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 52, 55)
    canvas.paste(_c17, (315, 6), _c17)
except Exception:
    pass
layout["icon_17"] = [315, 6, 367, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/18_icon_View_all.png
try:
    _c18 = get_crop(18, 288, 168)
    canvas.paste(_c18, (1152, 2792), _c18)
except Exception:
    pass
layout["View_all"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/19_text_Los_Angeles_CA.png
try:
    _c19 = get_crop(19, 461, 81)
    canvas.paste(_c19, (42, 131), _c19)
except Exception:
    pass
layout["Los_Angeles,_CA"] = [42, 131, 503, 212]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/20_text_Location.png
try:
    _c20 = get_crop(20, 235, 54)
    canvas.paste(_c20, (44, 382), _c20)
except Exception:
    pass
layout["Location"] = [44, 382, 279, 436]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/21_text_Date.png
try:
    _c21 = get_crop(21, 140, 60)
    canvas.paste(_c21, (42, 775), _c21)
except Exception:
    pass
layout["Date"] = [42, 775, 182, 835]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/22_text_Clear.png
try:
    _c22 = get_crop(22, 264, 149)
    canvas.paste(_c22, (1176, 730), _c22)
except Exception:
    pass
layout["Clear"] = [1176, 730, 1440, 879]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/23_text_Tomorrow.png
try:
    _c23 = get_crop(23, 448, 149)
    canvas.paste(_c23, (496, 901), _c23)
except Exception:
    pass
layout["Tomorrow"] = [496, 901, 944, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/24_text_Weekend.png
try:
    _c24 = get_crop(24, 448, 149)
    canvas.paste(_c24, (944, 901), _c24)
except Exception:
    pass
layout["Weekend"] = [944, 901, 1392, 1050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/25_text_Set_custom_date.png
try:
    _c25 = get_crop(25, 492, 149)
    canvas.paste(_c25, (474, 1052), _c25)
except Exception:
    pass
layout["Set_custom_date"] = [474, 1052, 966, 1201]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/26_text_Just_announced.png
try:
    _c26 = get_crop(26, 412, 54)
    canvas.paste(_c26, (42, 1691), _c26)
except Exception:
    pass
layout["Just_announced"] = [42, 1691, 454, 1745]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/27_text_View_all.png
try:
    _c27 = get_crop(27, 165, 43)
    canvas.paste(_c27, (1227, 1699), _c27)
except Exception:
    pass
layout["View_all"] = [1227, 1699, 1392, 1742]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/28_text_S46.png
try:
    _c28 = get_crop(28, 119, 52)
    canvas.paste(_c28, (95, 2037), _c28)
except Exception:
    pass
layout["S46+"] = [95, 2037, 214, 2089]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/29_text_Andrew_Schulz.png
try:
    _c29 = get_crop(29, 321, 52)
    canvas.paste(_c29, (46, 2162), _c29)
except Exception:
    pass
layout["Andrew_Schulz"] = [46, 2162, 367, 2214]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/30_text_Thu.png
try:
    _c30 = get_crop(30, 92, 45)
    canvas.paste(_c30, (45, 2235), _c30)
except Exception:
    pass
layout["Thu,"] = [45, 2235, 137, 2280]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/31_text_9_7.30_PM.png
try:
    _c31 = get_crop(31, 214, 49)
    canvas.paste(_c31, (234, 2232), _c31)
except Exception:
    pass
layout["9,7.30_PM"] = [234, 2232, 448, 2281]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/32_text_Sports.png
try:
    _c32 = get_crop(32, 179, 68)
    canvas.paste(_c32, (41, 2446), _c32)
except Exception:
    pass
layout["Sports"] = [41, 2446, 220, 2514]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/33_text_View_all.png
try:
    _c33 = get_crop(33, 165, 43)
    canvas.paste(_c33, (1227, 2452), _c33)
except Exception:
    pass
layout["View_all"] = [1227, 2452, 1392, 2495]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/34_text_55.png
try:
    _c34 = get_crop(34, 133, 145)
    canvas.paste(_c34, (562, 2641), _c34)
except Exception:
    pass
layout["55"] = [562, 2641, 695, 2786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/68e3462c14734440a7ace3fed432a10d/step_06_2024_3_20_16_29_68e3462c14734440a7ace3fed432a10d-9/35_clickable_Location.png
try:
    _c35 = get_crop(35, 1440, 937)
    canvas.paste(_c35, (0, 312), _c35)
except Exception:
    pass
layout["Location"] = [0, 312, 1440, 1249]
