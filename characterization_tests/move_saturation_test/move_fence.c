#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define ITERS 2000000UL
#define MAXK 256

// Ultra-hard-fenced MOV
#define DO_MOV(i)                                                      \
    asm volatile("lfence" ::: "memory");                               \
    asm volatile("mov %1, %0" : "+r"(a##i) : "r"(b##i));               \
    asm volatile("lfence" ::: "memory");

// Register declarations
#define DECL_REG(i) register uint64_t a##i = i, b##i = (i + 1);

// Emit K serialized MOVs using a fall-through switch
#define GEN_MOVS(K)                        \
    switch (K) {                            \
        case 256: DO_MOV(255);              \
        case 255: DO_MOV(254);              \
        case 254: DO_MOV(253);              \
        case 253: DO_MOV(252);              \
        case 252: DO_MOV(251);              \
        case 251: DO_MOV(250);              \
        case 250: DO_MOV(249);              \
        case 249: DO_MOV(248);              \
        case 248: DO_MOV(247);              \
        case 247: DO_MOV(246);              \
        case 246: DO_MOV(245);              \
        case 245: DO_MOV(244);              \
        case 244: DO_MOV(243);              \
        case 243: DO_MOV(242);              \
        case 242: DO_MOV(241);              \
        case 241: DO_MOV(240);              \
        case 240: DO_MOV(239);              \
        case 239: DO_MOV(238);              \
        case 238: DO_MOV(237);              \
        case 237: DO_MOV(236);              \
        case 236: DO_MOV(235);              \
        case 235: DO_MOV(234);              \
        case 234: DO_MOV(233);              \
        case 233: DO_MOV(232);              \
        case 232: DO_MOV(231);              \
        case 231: DO_MOV(230);              \
        case 230: DO_MOV(229);              \
        case 229: DO_MOV(228);              \
        case 228: DO_MOV(227);              \
        case 227: DO_MOV(226);              \
        case 226: DO_MOV(225);              \
        case 225: DO_MOV(224);              \
        case 224: DO_MOV(223);              \
        case 223: DO_MOV(222);              \
        case 222: DO_MOV(221);              \
        case 221: DO_MOV(220);              \
        case 220: DO_MOV(219);              \
        case 219: DO_MOV(218);              \
        case 218: DO_MOV(217);              \
        case 217: DO_MOV(216);              \
        case 216: DO_MOV(215);              \
        case 215: DO_MOV(214);              \
        case 214: DO_MOV(213);              \
        case 213: DO_MOV(212);              \
        case 212: DO_MOV(211);              \
        case 211: DO_MOV(210);              \
        case 210: DO_MOV(209);              \
        case 209: DO_MOV(208);              \
        case 208: DO_MOV(207);              \
        case 207: DO_MOV(206);              \
        case 206: DO_MOV(205);              \
        case 205: DO_MOV(204);              \
        case 204: DO_MOV(203);              \
        case 203: DO_MOV(202);              \
        case 202: DO_MOV(201);              \
        case 201: DO_MOV(200);              \
        case 200: DO_MOV(199);              \
        case 199: DO_MOV(198);              \
        case 198: DO_MOV(197);              \
        case 197: DO_MOV(196);              \
        case 196: DO_MOV(195);              \
        case 195: DO_MOV(194);              \
        case 194: DO_MOV(193);              \
        case 193: DO_MOV(192);              \
        case 192: DO_MOV(191);              \
        case 191: DO_MOV(190);              \
        case 190: DO_MOV(189);              \
        case 189: DO_MOV(188);              \
        case 188: DO_MOV(187);              \
        case 187: DO_MOV(186);              \
        case 186: DO_MOV(185);              \
        case 185: DO_MOV(184);              \
        case 184: DO_MOV(183);              \
        case 183: DO_MOV(182);              \
        case 182: DO_MOV(181);              \
        case 181: DO_MOV(180);              \
        case 180: DO_MOV(179);              \
        case 179: DO_MOV(178);              \
        case 178: DO_MOV(177);              \
        case 177: DO_MOV(176);              \
        case 176: DO_MOV(175);              \
        case 175: DO_MOV(174);              \
        case 174: DO_MOV(173);              \
        case 173: DO_MOV(172);              \
        case 172: DO_MOV(171);              \
        case 171: DO_MOV(170);              \
        case 170: DO_MOV(169);              \
        case 169: DO_MOV(168);              \
        case 168: DO_MOV(167);              \
        case 167: DO_MOV(166);              \
        case 166: DO_MOV(165);              \
        case 165: DO_MOV(164);              \
        case 164: DO_MOV(163);              \
        case 163: DO_MOV(162);              \
        case 162: DO_MOV(161);              \
        case 161: DO_MOV(160);              \
        case 160: DO_MOV(159);              \
        case 159: DO_MOV(158);              \
        case 158: DO_MOV(157);              \
        case 157: DO_MOV(156);              \
        case 156: DO_MOV(155);              \
        case 155: DO_MOV(154);              \
        case 154: DO_MOV(153);              \
        case 153: DO_MOV(152);              \
        case 152: DO_MOV(151);              \
        case 151: DO_MOV(150);              \
        case 150: DO_MOV(149);              \
        case 149: DO_MOV(148);              \
        case 148: DO_MOV(147);              \
        case 147: DO_MOV(146);              \
        case 146: DO_MOV(145);              \
        case 145: DO_MOV(144);              \
        case 144: DO_MOV(143);              \
        case 143: DO_MOV(142);              \
        case 142: DO_MOV(141);              \
        case 141: DO_MOV(140);              \
        case 140: DO_MOV(139);              \
        case 139: DO_MOV(138);              \
        case 138: DO_MOV(137);              \
        case 137: DO_MOV(136);              \
        case 136: DO_MOV(135);              \
        case 135: DO_MOV(134);              \
        case 134: DO_MOV(133);              \
        case 133: DO_MOV(132);              \
        case 132: DO_MOV(131);              \
        case 131: DO_MOV(130);              \
        case 130: DO_MOV(129);              \
        case 129: DO_MOV(128);              \
        case 128: DO_MOV(127);              \
        case 127: DO_MOV(126);              \
        case 126: DO_MOV(125);              \
        case 125: DO_MOV(124);              \
        case 124: DO_MOV(123);              \
        case 123: DO_MOV(122);              \
        case 122: DO_MOV(121);              \
        case 121: DO_MOV(120);              \
        case 120: DO_MOV(119);              \
        case 119: DO_MOV(118);              \
        case 118: DO_MOV(117);              \
        case 117: DO_MOV(116);              \
        case 116: DO_MOV(115);              \
        case 115: DO_MOV(114);              \
        case 114: DO_MOV(113);              \
        case 113: DO_MOV(112);              \
        case 112: DO_MOV(111);              \
        case 111: DO_MOV(110);              \
        case 110: DO_MOV(109);              \
        case 109: DO_MOV(108);              \
        case 108: DO_MOV(107);              \
        case 107: DO_MOV(106);              \
        case 106: DO_MOV(105);              \
        case 105: DO_MOV(104);              \
        case 104: DO_MOV(103);              \
        case 103: DO_MOV(102);              \
        case 102: DO_MOV(101);              \
        case 101: DO_MOV(100);              \
        case 100: DO_MOV(99);               \
        case 99:  DO_MOV(98);               \
        case 98:  DO_MOV(97);               \
        case 97:  DO_MOV(96);               \
        case 96:  DO_MOV(95);               \
        case 95:  DO_MOV(94);               \
        case 94:  DO_MOV(93);               \
        case 93:  DO_MOV(92);               \
        case 92:  DO_MOV(91);               \
        case 91:  DO_MOV(90);               \
        case 90:  DO_MOV(89);               \
        case 89:  DO_MOV(88);               \
        case 88:  DO_MOV(87);               \
        case 87:  DO_MOV(86);               \
        case 86:  DO_MOV(85);               \
        case 85:  DO_MOV(84);               \
        case 84:  DO_MOV(83);               \
        case 83:  DO_MOV(82);               \
        case 82:  DO_MOV(81);               \
        case 81:  DO_MOV(80);               \
        case 80:  DO_MOV(79);               \
        case 79:  DO_MOV(78);               \
        case 78:  DO_MOV(77);               \
        case 77:  DO_MOV(76);               \
        case 76:  DO_MOV(75);               \
        case 75:  DO_MOV(74);               \
        case 74:  DO_MOV(73);               \
        case 73:  DO_MOV(72);               \
        case 72:  DO_MOV(71);               \
        case 71:  DO_MOV(70);               \
        case 70:  DO_MOV(69);               \
        case 69:  DO_MOV(68);               \
        case 68:  DO_MOV(67);               \
        case 67:  DO_MOV(66);               \
        case 66:  DO_MOV(65);               \
        case 65:  DO_MOV(64);               \
        case 64:  DO_MOV(63);               \
        case 63:  DO_MOV(62);               \
        case 62:  DO_MOV(61);               \
        case 61:  DO_MOV(60);               \
        case 60:  DO_MOV(59);               \
        case 59:  DO_MOV(58);               \
        case 58:  DO_MOV(57);               \
        case 57:  DO_MOV(56);               \
        case 56:  DO_MOV(55);               \
        case 55:  DO_MOV(54);               \
        case 54:  DO_MOV(53);               \
        case 53:  DO_MOV(52);               \
        case 52:  DO_MOV(51);               \
        case 51:  DO_MOV(50);               \
        case 50:  DO_MOV(49);               \
        case 49:  DO_MOV(48);               \
        case 48:  DO_MOV(47);               \
        case 47:  DO_MOV(46);               \
        case 46:  DO_MOV(45);               \
        case 45:  DO_MOV(44);               \
        case 44:  DO_MOV(43);               \
        case 43:  DO_MOV(42);               \
        case 42:  DO_MOV(41);               \
        case 41:  DO_MOV(40);               \
        case 40:  DO_MOV(39);               \
        case 39:  DO_MOV(38);               \
        case 38:  DO_MOV(37);               \
        case 37:  DO_MOV(36);               \
        case 36:  DO_MOV(35);               \
        case 35:  DO_MOV(34);               \
        case 34:  DO_MOV(33);               \
        case 33:  DO_MOV(32);               \
        case 32:  DO_MOV(31);               \
        case 31:  DO_MOV(30);               \
        case 30:  DO_MOV(29);               \
        case 29:  DO_MOV(28);               \
        case 28:  DO_MOV(27);               \
        case 27:  DO_MOV(26);               \
        case 26:  DO_MOV(25);               \
        case 25:  DO_MOV(24);               \
        case 24:  DO_MOV(23);               \
        case 23:  DO_MOV(22);               \
        case 22:  DO_MOV(21);               \
        case 21:  DO_MOV(20);               \
        case 20:  DO_MOV(19);               \
        case 19:  DO_MOV(18);               \
        case 18:  DO_MOV(17);               \
        case 17:  DO_MOV(16);               \
        case 16:  DO_MOV(15);               \
        case 15:  DO_MOV(14);               \
        case 14:  DO_MOV(13);               \
        case 13:  DO_MOV(12);               \
        case 12:  DO_MOV(11);               \
        case 11:  DO_MOV(10);               \
        case 10:  DO_MOV(9);                \
        case 9:   DO_MOV(8);                \
        case 8:   DO_MOV(7);                \
        case 7:   DO_MOV(6);                \
        case 6:   DO_MOV(5);                \
        case 5:   DO_MOV(4);                \
        case 4:   DO_MOV(3);                \
        case 3:   DO_MOV(2);                \
        case 2:   DO_MOV(1);                \
        case 1:   DO_MOV(0);                \
        default:                              \
            break;                             \
    }

__attribute__((noinline))
void mov_elim_sat(int K) {
    if (K < 1 || K > MAXK) return;

    // Declare 256 register pairs
    #define X(i) DECL_REG(i)
    X(0);   X(1);   X(2);   X(3);   X(4);   X(5);   X(6);   X(7);
    X(8);   X(9);   X(10);  X(11);  X(12);  X(13);  X(14);  X(15);
    X(16);  X(17);  X(18);  X(19);  X(20);  X(21);  X(22);  X(23);
    X(24);  X(25);  X(26);  X(27);  X(28);  X(29);  X(30);  X(31);
    X(32);  X(33);  X(34);  X(35);  X(36);  X(37);  X(38);  X(39);
    X(40);  X(41);  X(42);  X(43);  X(44);  X(45);  X(46);  X(47);
    X(48);  X(49);  X(50);  X(51);  X(52);  X(53);  X(54);  X(55);
    X(56);  X(57);  X(58);  X(59);  X(60);  X(61);  X(62);  X(63);
    X(64);  X(65);  X(66);  X(67);  X(68);  X(69);  X(70);  X(71);
    X(72);  X(73);  X(74);  X(75);  X(76);  X(77);  X(78);  X(79);
    X(80);  X(81);  X(82);  X(83);  X(84);  X(85);  X(86);  X(87);
    X(88);  X(89);  X(90);  X(91);  X(92);  X(93);  X(94);  X(95);
    X(96);  X(97);  X(98);  X(99);  X(100); X(101); X(102); X(103);
    X(104); X(105); X(106); X(107); X(108); X(109); X(110); X(111);
    X(112); X(113); X(114); X(115); X(116); X(117); X(118); X(119);
    X(120); X(121); X(122); X(123); X(124); X(125); X(126); X(127);
    X(128); X(129); X(130); X(131); X(132); X(133); X(134); X(135);
    X(136); X(137); X(138); X(139); X(140); X(141); X(142); X(143);
    X(144); X(145); X(146); X(147); X(148); X(149); X(150); X(151);
    X(152); X(153); X(154); X(155); X(156); X(157); X(158); X(159);
    X(160); X(161); X(162); X(163); X(164); X(165); X(166); X(167);
    X(168); X(169); X(170); X(171); X(172); X(173); X(174); X(175);
    X(176); X(177); X(178); X(179); X(180); X(181); X(182); X(183);
    X(184); X(185); X(186); X(187); X(188); X(189); X(190); X(191);
    X(192); X(193); X(194); X(195); X(196); X(197); X(198); X(199);
    X(200); X(201); X(202); X(203); X(204); X(205); X(206); X(207);
    X(208); X(209); X(210); X(211); X(212); X(213); X(214); X(215);
    X(216); X(217); X(218); X(219); X(220); X(221); X(222); X(223);
    X(224); X(225); X(226); X(227); X(228); X(229); X(230); X(231);
    X(232); X(233); X(234); X(235); X(236); X(237); X(238); X(239);
    X(240); X(241); X(242); X(243); X(244); X(245); X(246); X(247);
    X(248); X(249); X(250); X(251); X(252); X(253); X(254); X(255);
    #undef X

    for (uint64_t i = 0; i < ITERS; i++) {
        GEN_MOVS(K);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <K 1-256>\n", argv[0]);
        return 1;
    }

    int K = atoi(argv[1]);
    if (K < 1 || K > MAXK) {
        fprintf(stderr, "K must be 1-%d\n", MAXK);
        return 1;
    }

    mov_elim_sat(K);
    return 0;
}
