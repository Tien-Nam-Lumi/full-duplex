class FakeLLM:
    def generate_response(self, command: str) -> str:
        command = (command or "").lower()
        if "bật đèn" in command:
            return "Đã bật đèn phòng khách."
        return "Xin lỗi, tôi không hiểu."

