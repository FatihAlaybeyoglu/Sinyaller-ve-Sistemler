# signals_and_systems_homework_2.py
# Açıklamalı sürüm - @brief, @param, @return formatında (kod değiştirilmedi)

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QHBoxLayout, QDialog, QGroupBox, QGridLayout, QComboBox, QFileDialog, QSpinBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages

# ----------- YENİ: ZoomDialog'a yakınlaştırma eklendi -----------
class ZoomDialog(QDialog):
    """
    @brief Çizilen sinyali ayrı bir pencerede yakınlaştırma/uzaklaştırma ve kaydırma ile incelemeyi sağlar.
    @param title  : Pencere başlığı
    @param t      : Zaman ekseni (numpy.ndarray)
    @param signal : Sinyal değerleri (numpy.ndarray)
    @param period : (Opsiyonel) Sinyalin periyodu; None ise t’den türetilir.
    @return       : Kullanıcı etkileşimi sağlayan QDialog
    """
    def __init__(self, title, t, signal, period=None):
        super().__init__()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.original_t = t
        self.signal = signal
        self.period = period if period is not None else (t[-1] - t[0])
        self.n_periods = (t[-1] - t[0]) / self.period if self.period else 1

        self.current_start = t[0]
        self.current_end = t[-1]
        self.zoom_factor = 2

        layout = QVBoxLayout()

        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_title(title, color='white')

        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.coords_label = QLabel("X: -, Y: -")
        self.coords_label.setFont(QFont("Courier", 10))
        self.coords_label.setStyleSheet("color: white; padding: 4px;")
        layout.addWidget(self.coords_label, alignment=Qt.AlignTop | Qt.AlignLeft)

        # --- Zoom ve Kaydırma Butonları ---
        button_layout = QHBoxLayout()
        self.left_btn = QPushButton("◀ Kaydır")
        self.left_btn.clicked.connect(self.shift_left)
        self.right_btn = QPushButton("Kaydır ▶")
        self.right_btn.clicked.connect(self.shift_right)
        self.zoom_in_btn = QPushButton("Yakınlaştır (+)")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn = QPushButton("Uzaklaştır (-)")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        for btn in (self.left_btn, self.right_btn, self.zoom_in_btn, self.zoom_out_btn):
            btn.setStyleSheet("background-color: #444; color: white; margin:2px; padding:4px 10px; border-radius:5px;")
            button_layout.addWidget(btn)
        layout.addLayout(button_layout)
        # Klavyeden de kontrol için:
        self.setFocusPolicy(Qt.StrongFocus)

        def on_motion(event):
            if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                self.coords_label.setText(f"X: {event.xdata:.3f}, Y: {event.ydata:.3f}")

        self.canvas.mpl_connect("motion_notify_event", on_motion)

        self.setLayout(layout)
        self.resize(1200, 540)
        self.redraw()

    def redraw(self):
        """@brief Aktif pencere aralığına göre grafiği yeniden çizer. @return None"""
        self.ax.clear()
        mask = (self.original_t >= self.current_start) & (self.original_t <= self.current_end)
        t = self.original_t[mask]
        sig = self.signal[mask]
        self.ax.plot(t, sig, color='#00d084')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_title(self.windowTitle(), color='white')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.canvas.draw()

    def zoom_in(self):
        """@brief Merkez etrafında yakınlaştırır. @return None"""
        mid = (self.current_start + self.current_end) / 2
        width = (self.current_end - self.current_start) / self.zoom_factor
        self.current_start = max(self.original_t[0], mid - width / 2)
        self.current_end = min(self.original_t[-1], mid + width / 2)
        self.redraw()

    def zoom_out(self):
        """@brief Merkez etrafında uzaklaştırır. @return None"""
        mid = (self.current_start + self.current_end) / 2
        width = (self.current_end - self.current_start) * self.zoom_factor
        total_width = self.original_t[-1] - self.original_t[0]
        new_start = max(self.original_t[0], mid - width / 2)
        new_end = min(self.original_t[-1], mid + width / 2)
        if new_end - new_start > total_width:
            new_start = self.original_t[0]
            new_end = self.original_t[-1]
        self.current_start = new_start
        self.current_end = new_end
        self.redraw()

    def shift_left(self):
        """@brief Görünümü sola kaydırır. @return None"""
        width = self.current_end - self.current_start
        shift = width * 0.2  # pencerenin %20'si kadar kaydır
        new_start = self.current_start - shift
        new_end = self.current_end - shift
        if new_start < self.original_t[0]:
            new_start = self.original_t[0]
            new_end = new_start + width
        self.current_start = new_start
        self.current_end = new_end
        self.redraw()

    def shift_right(self):
        """@brief Görünümü sağa kaydırır. @return None"""
        width = self.current_end - self.current_start
        shift = width * 0.2  # pencerenin %20'si kadar kaydır
        new_end = self.current_end + shift
        new_start = self.current_start + shift
        if new_end > self.original_t[-1]:
            new_end = self.original_t[-1]
            new_start = new_end - width
        self.current_start = new_start
        self.current_end = new_end
        self.redraw()

    def keyPressEvent(self, event):
        """
        @brief Klavye kısayolları ile kontrol (← → - +).
        @param event: QKeyEvent
        @return None
        """
        if event.key() == Qt.Key_Left:
            self.shift_left()
        elif event.key() == Qt.Key_Right:
            self.shift_right()
        elif event.key() == Qt.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_in()
        else:
            super().keyPressEvent(event)


#----------- YENİ: Periyot seçimi için yardımcı fonksiyon -----------
def get_t_with_period(frequency, n_periods, n_points=1000):
    """
    @brief Verilen frekans ve periyot sayısına göre zaman ekseni üretir.
    @param frequency: Frekans (Hz). 0 ise T=1 varsayılır.
    @param n_periods: Kaç periyot gösterileceği (int)
    @param n_points : Örnek sayısı (int, varsayılan 1000)
    @return t (np.ndarray), period (float)
    """
    if frequency == 0:
        return np.linspace(0, 1, n_points), 1  # periyot 1 kabul
    period = 1 / frequency
    t_end = n_periods * period
    return np.linspace(0, t_end, n_points), period

# ----------- PlotCanvas'ta da periyot parametresi eklendi -----------
class PlotCanvas(FigureCanvas):
    """
    @brief Dört ayrı eksende üç bireysel sinyal ve bir toplam sinyal çizer.
    @param parent: Üst widget
    @return FigureCanvas türevi çizim yüzeyi
    """
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'hspace': 0.8})
        super().__init__(fig)
        self.setParent(parent)
        self.signals = [(None, None, None)] * 4  # (t, y, period)
        fig.patch.set_facecolor('#1e1e1e')
        for axis in self.ax:
            axis.set_facecolor('#1e1e1e')
            axis.tick_params(colors='white')
            for spine in axis.spines.values():
                spine.set_color('white')
            axis.title.set_color('white')
            axis.yaxis.label.set_color('white')
            axis.xaxis.label.set_color('white')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95)
        self.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        """@brief Eksenlerden birine tıklanınca ZoomDialog açar. @param event: mpl event"""
        for i, ax in enumerate(self.ax):
            if event.inaxes == ax and self.signals[i][0] is not None:
                t, signal, period = self.signals[i]
                dlg = ZoomDialog(ax.get_title(), t, signal, period)
                dlg.exec_()

    def plot_individual(self, signal_index, t, signal, period, label=None):
        """
        @brief Tek bir sinyali ilgili eksende çizer.
        @param signal_index: 0..2 arası indeks
        @param t: zaman ekseni
        @param signal: sinyal değerleri
        @param period: periyot
        @param label: başlık etiketi
        """
        self.ax[signal_index].clear()
        self.ax[signal_index].plot(t, signal, color='cyan')
        self.signals[signal_index] = (t, signal, period)
        title = label if label else f"Sinyal {signal_index + 1}"
        self.ax[signal_index].set_title(title + f" | Periyot: {period:.3g}", color='white', pad=15)
        self.draw()

    def plot_total(self, total, t=None, period=None, label="Toplam (Sentez) Sinyal"):
        """
        @brief Toplam (sentez) sinyali çizer.
        @param total: toplam sinyal
        @param t: zaman ekseni (opsiyonel)
        @param period: periyot (opsiyonel)
        @param label: başlık
        """
        self.ax[3].clear()
        if t is None:
            for t_candidate, _, _ in self.signals:
                if t_candidate is not None:
                    t = t_candidate
                    break
        if period is None:
            period = t[1] - t[0] if t is not None else 1
        self.ax[3].plot(t, total, color='magenta')
        self.ax[3].set_title(label + f" | Periyot: {period:.3g}", color='white', pad=15)
        self.signals[3] = (t, total, period)
        self.draw()

    def clear_all(self):
        """@brief Tüm eksenleri ve kayıtlı sinyalleri temizler. @return None"""
        for ax in self.ax:
            ax.clear()
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.title.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
        self.signals = [(None, None, None)] * 4
        self.draw()

    def save_all_plots_as_pdf(self):
        """@brief Tüm grafikleri tek bir PDF dosyasına kaydeder. @return None"""
        path, _ = QFileDialog.getSaveFileName(None, "PDF Kaydet", "grafikler.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        with PdfPages(path) as pdf:
            for i, (t, signal, period) in enumerate(self.signals):
                if signal is None or t is None:
                    continue
                fig, ax = plt.subplots(figsize=(14, 5))
                fig.patch.set_facecolor('#1e1e1e')
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
                ax.title.set_color('white')
                label = f"Sinyal {i+1}" if i < 3 else "Toplam (Sentez) Sinyal"
                ax.set_title(label + f" | Periyot: {period:.3g}", color='white')
                color = 'cyan' if i < 3 else 'magenta'
                ax.plot(t, signal, color=color)
                pdf.savefig(fig)
                plt.close(fig)


class FourierGUI(QWidget):
    """
    @brief Kullanıcı arayüzü: üç sinyal tanımlayıp çizer, toplam sinyali üretir,
           ayrıca Fourier serisi bileşenleri için arayüz sunar.
    @return QWidget türevi ana pencere
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Sinyaller ve Sistemler Ödev 2')
        self.resize(1920, 1080)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.layout = QVBoxLayout()
        self.inputs = []
        self.signals = [None, None, None]
        self.types = []
        self.last_was_fourier = False
        self.period_inputs = []

        group_box = QGroupBox("Sinyal Parametreleri (θ derece cinsindendir)")
        group_box.setStyleSheet("color: white; background-color: #2e2e2e; margin: 10px; padding: 10px;")
        grid = QGridLayout()
        labels = ['A1', 'f1', 'θ1 (°)', 'A2', 'f2', 'θ2 (°)', 'A3', 'f3', 'θ3 (°)']

        for i in range(3):
            combo = QComboBox()
            combo.addItems(["sin", "cos"])
            combo.setStyleSheet("margin: 5px; padding: 5px; background-color: #444; color: white; border: 1px solid #888;")
            self.types.append(combo)
            grid.addWidget(combo, i, 0)

            label_a = QLabel(labels[i*3])
            input_a = QLineEdit('0')
            input_a.setStyleSheet("margin: 5px; padding: 5px; background-color: #222; color: white;")
            grid.addWidget(label_a, i, 1)
            grid.addWidget(input_a, i, 2)
            self.inputs.append(input_a)

            label_f = QLabel(labels[i*3 + 1])
            input_f = QLineEdit('0')
            input_f.setStyleSheet("margin: 5px; padding: 5px; background-color: #222; color: white;")
            grid.addWidget(label_f, i, 3)
            grid.addWidget(input_f, i, 4)
            self.inputs.append(input_f)

            label_theta = QLabel(labels[i*3 + 2])
            input_theta = QLineEdit('0')
            input_theta.setStyleSheet("margin: 5px; padding: 5px; background-color: #222; color: white;")
            grid.addWidget(label_theta, i, 5)
            grid.addWidget(input_theta, i, 6)
            self.inputs.append(input_theta)

            # ----------- YENİ: Periyot sayısı kutusu -----------
            period_label = QLabel("Periyot sayısı:")
            period_box = QLineEdit('3')
            period_box.setStyleSheet("background-color: #444; color: white;")
            grid.addWidget(period_label, i, 7)
            grid.addWidget(period_box, i, 8)
            self.period_inputs.append(period_box)

            btn = QPushButton(f"Sinyal {i+1} Çiz")
            btn.clicked.connect(lambda _, idx=i: self.draw_single(idx))
            btn.setStyleSheet(
                "background-color: #ff9100; color: white; font-weight: bold; border-radius: 5px; padding: 6px 12px;"
            )
            grid.addWidget(btn, i, 9)

        group_box.setLayout(grid)
        self.layout.addWidget(group_box)

        self.canvas = PlotCanvas(self)
        self.layout.addWidget(self.canvas)

        button_row = QHBoxLayout()

        self.fourier_button = QPushButton('Fourier Serisi')
        self.fourier_button.clicked.connect(self.open_fourier_dialog)
        self.fourier_button.setStyleSheet("background-color: #0080ff; color: white; font-weight: bold; margin: 10px; padding: 10px 20px; border-radius: 6px;")
        button_row.addWidget(self.fourier_button)

        self.pdf_button = QPushButton('PDF Kaydet')
        self.pdf_button.clicked.connect(self.canvas.save_all_plots_as_pdf)
        self.pdf_button.setStyleSheet("background-color: #00c853; color: white; font-weight: bold; margin: 10px; padding: 10px 20px; border-radius: 6px;")
        button_row.addWidget(self.pdf_button)

        self.reset_button = QPushButton('Sıfırla')
        self.reset_button.clicked.connect(self.reset_all)
        self.reset_button.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; margin: 10px; padding: 10px 20px; border-radius: 6px;")
        button_row.addWidget(self.reset_button)

        self.layout.addLayout(button_row)
        self.setLayout(self.layout)

    def reset_all(self):
        """@brief Tüm çizimleri sıfırlar ve dahili durumları temizler. @return None"""
        self.canvas.clear_all()
        self.signals = [None, None, None]
        self.last_was_fourier = False

    def draw_single(self, idx):
        """
        @brief Tek bir sinyali kullanıcı girişlerine göre üretir, çizer ve sentez toplamını günceller.
        @param idx: Sinyal indeksi (0,1,2)
        @return None
        """
        try:
            if self.last_was_fourier:
                self.reset_all()

            # 1) Seçilen sinyalin parametrelerini oku ve üret
            A = float(self.inputs[idx * 3].text())
            f = float(self.inputs[idx * 3 + 1].text())
            theta = np.radians(float(self.inputs[idx * 3 + 2].text()))
            typ = self.types[idx].currentText()
            try:
                n_periods = int(self.period_inputs[idx].text())
                if n_periods < 1:
                    n_periods = 1
            except Exception:
                n_periods = 1
            period = 1 / f if f != 0 else 1
            t = np.linspace(0, n_periods * period, 1000)
            if typ == "cos":
                y = A * np.cos(2 * np.pi * f * t + theta)
            else:
                y = A * np.sin(2 * np.pi * f * t + theta)

            # Yalnızca ilgili sinyali güncelle
            self.signals[idx] = {
                'A': A, 'f': f, 'theta': theta, 'typ': typ, 'period': period, 't': t, 'y': y
            }

            # Sadece ilgili sinyali çiz
            self.canvas.plot_individual(idx, t, y, period, label=f"Sinyal {idx + 1}")

            # --- Sentez için: EN BÜYÜK frekansı bul, oraya göre ortak t ekseni oluştur ---
            valid_signals = [sig for sig in self.signals if sig is not None and sig['f'] > 0]
            if not valid_signals:
                # Hiç sinyal yoksa sentez çizme
                self.canvas.plot_total(np.zeros(1000), np.linspace(0, 1, 1000), 1, label="Toplam (Sentez) Sinyal")
                return

            max_f = max(sig['f'] for sig in valid_signals)
            sentez_period = 1 / max_f if max_f != 0 else 1
            sentez_n_periods = 3
            sentez_t = np.linspace(0, sentez_n_periods * sentez_period, 1000)

            # Her tanımlı sinyali bu sentez_t üzerinde yeniden oluştur
            sentez_signals = []
            for sig in self.signals:
                if sig is not None:
                    f = sig['f']
                    A = sig['A']
                    theta = sig['theta']
                    typ = sig['typ']
                    if typ == "cos":
                        y_sentez = A * np.cos(2 * np.pi * f * sentez_t + theta)
                    else:
                        y_sentez = A * np.sin(2 * np.pi * f * sentez_t + theta)
                    sentez_signals.append(y_sentez)
            total = np.sum(sentez_signals, axis=0)
            self.canvas.plot_total(total, sentez_t, sentez_period, label="Toplam (Sentez) Sinyal")
            self.last_was_fourier = False

        except Exception as e:
            print("Hata:", e)

    def open_fourier_dialog(self):
        """
        @brief Fourier serisi katsayılarını girmek ve sinyali çizmek için diyalog açar.
        @return None
        """
        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Fourier Serisi Katsayı Arayüzü (k = 1, 2, 3)")
        dialog.setStyleSheet("background-color: #2e2e2e; color: white; padding: 10px;")
        layout = QVBoxLayout()

        self.a0_input = QLineEdit("0")
        layout.addWidget(QLabel("a₀: "))
        layout.addWidget(self.a0_input)

        self.ak_inputs = []
        self.bk_inputs = []
        for k in range(1, 4):
            layout.addWidget(QLabel(f"a{k}: "))
            ak = QLineEdit("0")
            self.ak_inputs.append(ak)
            layout.addWidget(ak)

            layout.addWidget(QLabel(f"b{k}: "))
            bk = QLineEdit("0")
            self.bk_inputs.append(bk)
            layout.addWidget(bk)

        self.T_input = QLineEdit("1")
        layout.addWidget(QLabel("Periyot (T): "))
        layout.addWidget(self.T_input)

        per_label = QLabel("Kaç Periyot Gösterilsin?")
        self.n_periods_input = QLineEdit('3')
        self.n_periods_input.setStyleSheet("background-color: #444; color: white;")
        layout.addWidget(per_label)
        layout.addWidget(self.n_periods_input)

        plot_btn = QPushButton("Fourier Sinyalini Çiz")
        plot_btn.setStyleSheet("background-color: #0080ff; color: white; font-weight: bold; margin: 10px; padding: 10px 20px; border-radius: 6px;")
        plot_btn.clicked.connect(self.plot_fourier_signal)
        layout.addWidget(plot_btn)

        close_btn = QPushButton("Kapat")
        close_btn.setStyleSheet("background-color: #00c853; color: white; font-weight: bold; margin: 10px; padding: 10px 20px; border-radius: 6px;")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def plot_fourier_signal(self):
        """
        @brief Girilen a0, (ak,bk) ve T periyoduna göre üç harmonik + DC toplamını çizer.
        @return None
        """
        try:
            T = float(self.T_input.text())
            try:
                n_periods = int(self.n_periods_input.text())
                if n_periods < 1:
                    n_periods = 1
            except Exception:
                n_periods = 1

            t = np.linspace(0, n_periods * T, 1000)
            w0 = 2 * np.pi / T
            signals = []

            for k in range(1, 4):
                ak = float(self.ak_inputs[k - 1].text())
                bk = float(self.bk_inputs[k - 1].text())
                sk = ak * np.cos(k * w0 * t) + bk * np.sin(k * w0 * t)
                signals.append(sk)
                self.canvas.plot_individual(k - 1, t, sk, T, label=f"Harmonik {k} (a{k} cos + b{k} sin)")

            a0 = float(self.a0_input.text())
            total = a0 * np.ones_like(t)
            for sk in signals:
                total += sk

            self.canvas.plot_total(total, t, T, label="Fourier Toplamı")
            self.last_was_fourier = True

        except ValueError:
            print("Geçerli sayılar girilmelidir.")


if __name__ == '__main__':
    # @brief Uygulama başlatıcı
    app = QApplication(sys.argv)
    window = FourierGUI()
    window.showMaximized()
    sys.exit(app.exec_())
