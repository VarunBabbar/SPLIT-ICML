#include "optimizer.hpp"

void Optimizer::diagnose_non_convergence(void) {
    diagnose_non_convergence(this->root);
    return;
}
bool Optimizer::diagnose_non_convergence(key_type const &key) {
    if (m_config.diagnostics == false) {
        return false;
    }

    vertex_accessor task;
    if (!m_graph.vertices.find(task, key)) {
        std::cout << "Missing a downward call:" << std::endl;
        std::cout << key.to_string() << std::endl;
        return true;
    }

    if (task->second.uncertainty() == 0 || task->second.lowerbound() >= task->second.upperscope()) {
        return false;
    }

    std::cout << "Non-Convergent Task" << std::endl;
    std::cout << task->second.capture_set().to_string() << std::endl;
    std::cout << task->second.inspect() << std::endl;

    unsigned int reasons = 0;
    bound_accessor bounds;
    m_graph.bounds.find(bounds, task->second.capture_set());
    for (bound_iterator iterator = bounds->second.begin(); iterator != bounds->second.end(); ++iterator) {
        int feature = std::get<0>(*iterator);
        bool ready = true;
        float lower = 0.0, upper = 0.0;
        for (int sign = -1; sign <= 1; sign += 2) {
            vertex_accessor child;
            child_accessor key;
            ready = ready &&
                    m_graph.children.find(key, std::make_pair(task->second.capture_set(), sign * (feature + 1))) &&
                    m_graph.vertices.find(child, key->second);
            if (ready) {
                lower += child->second.lowerbound();
                upper += child->second.upperbound();
            }
        }
        bool missing_signal = false;
        if (ready && (lower != std::get<1>(*iterator) || upper != std::get<2>(*iterator)) &&
            (lower > task->second.lowerbound() || upper < task->second.upperbound())) {
            missing_signal = true;
            std::get<1>(*iterator) = lower;
            std::get<2>(*iterator) = upper;
            std::cout << "Missing Signal:" << std::endl;
            std::cout << "Missing Signal From Feature: " << feature << std::endl;
        }

        float boundary = std::min(task->second.upperbound(), task->second.upperscope());
        if (std::get<1>(*iterator) + std::numeric_limits<float>::epsilon() > boundary) {
            continue;
        }
        if (std::get<1>(*iterator) == std::get<2>(*iterator)) {
            continue;
        }

        if (std::get<1>(*iterator) != task->second.lowerbound() &&
            std::get<2>(*iterator) != task->second.upperbound()) {
            continue;
        }

        ++reasons;

        std::cout << "Non-Convergent Feature: " << feature << ", Bounds: [" << std::get<1>(*iterator) << ", "
                  << std::get<2>(*iterator) << "]" << std::endl;

        {
            vertex_accessor child;
            child_accessor key;
            bool found = false;
            if (m_graph.children.find(key, std::make_pair(task->second.capture_set(), -(feature + 1)))) {
                float uncertainty = 0.0;
                vertex_accessor subtask;
                if (m_graph.vertices.find(subtask, key->second)) {
                    found = true;

                    if (task->second.capture_set().to_string() ==
                        "55 : "
                        "000000110100100101011101011101011111001001010100110110"
                        "0000000100"
                        "100100101001101011101011111001001010100110110011110011"
                        "0101100111"
                        "010101011111010111001001000110110100110001011010000010"
                        "1010101011"
                        "101010111001001000110110100111111111111111111111110111"
                        "1111011111"
                        "101101000111111100111111111010110111101110111110101111"
                        "1001001010"
                        "110110100111111111111111111111111111111111111111110101"
                        "0011111100"
                        "111111111111111111111111111111111111011110001001111110"
                        "0111111111"
                        "111111111101110101111101011110100100011011010011111011"
                        "1111111011"
                        "111111101111101011110100100011011010011111111111111111"
                        "1111111101"
                        "111111111100100101010011010011111111111111111111111111"
                        "1110101111"
                        "100100101010011010011111111111111111111010111110111011"
                        "1011111011"
                        "000111110011111111111111111111101010111110101111010010"
                        "0011011010"
                        "011111111111111111111111111111111111111011011101001001"
                        "0011111111"
                        "011111111111011111111111111101111101110011011001111111"
                        "1111111111"
                        "111111111110111011101111101100011111001111111111111111"
                        "1111111111"
                        "111111111110010010101101101001111111001011111011101011"
                        "1110111011"
                        "101111101110010110001111111111111010111010101011101010"
                        "1110010010"
                        "001101101001111111111111111111111111111111111111111111"
                        "1111011110"
                        "001111111111111111111111111111111011111001001000110110"
                        "1001111111"
                        "111111111111111111111011101110111110011011111000111111"
                        "1111111111"
                        "111111111011111111111111111001111011000111001111111011"
                        "1101011101"
                        "111101010111001001100110110100111111111111111111111111"
                        "1111111111"
                        "111111111111111011000111111111111111111111111111111111"
                        "1111111111"
                        "111111100100111111111101111101110101111101110110011011"
                        "0111001001"
                        "000111111111111111111111111111110111011100100100011011"
                        "0100111111"
                        "111111111111111111111111110111011111010101110100011111"
                        "1111111111"
                        "111110101111101110111011111011100100100011111111111110"
                        "1110111010"
                        "111110101011100100100011011010011111111111111111111111"
                        "1111111111"
                        "111101101100111011010011111111111111110111110111110111"
                        "0111011011"
                        "010101101100011111111111111111111111111111011111110110"
                        "1111110010"
                        "010011111111111111111011101110111011101110110110001101"
                        "1010011111"
                        "111111111111111111101101111111011101101010110010010111"
                        "1111111111"
                        "111111111111011101110111101111111010000010111111111111"
                        "1111111111"
                        "111011101110011101111111010010010111111111111011111110"
                        "1111111111"
                        "101011101101010110010001111111111111111111111111101110"
                        "1110111101"
                        "111010110010010111111011011111010111010110011101000110"
                        "1101110010"
                        "010010111111111110111101111111111101111111011011111100"
                        "1001001111"
                        "111111111111110101110100110111000110110111001001001011"
                        "1111111111"
                        "111111111111101110110001110110101011001001011111110111"
                        "1111111101"
                        "111101110111011110111111100000000111111111111111111011"
                        "1110100110"
                        "111001110110111011001001011111100101111101010101010001"
                        "0101000110"
                        "110111001001001011111111111111111111101111001010100011"
                        "0110111001"
                        "001000111111111111111111011111010011010100011011011100"
                        "1001001011"
                        "111101101111101011101010001010100011011011101100100101"
                        "1111111111"
                        "111111011011011011111010111011110111101100101111111111"
                        "1111111111"
                        "111011011111000111011110111100000011111111111111111111"
                        "1111010011"
                        "011100111011111110000000101111111111110111101100101000"
                        "1010000011"
                        "01101010110010010") {
                    }

                    if (task->second.capture_set().to_string() ==
                        "24 : "
                        "111111101100100100111001111111101110101110111001111111"
                        "1011101000"
                        "001010011111011111011101010110011111111011101011101100"
                        "0111111110"
                        "111010111011000111111110111010111011000111111110111010"
                        "1110110001"
                        "111111101110101110110001111111111110101110110001111101"
                        "1111011100"
                        "011010011111011111010101010110011111101110001000001010"
                        "0111111111"
                        "111010111011000111110111110111010100100111111111111111"
                        "1101111001"
                        "111111111111110101011001111111111111101110110001111111"
                        "1111111011"
                        "101100011011011111010101010110011111111111111011101100"
                        "0111111111"
                        "111111110110100111111111111111111011000111111111111111"
                        "1110110001"
                        "111111111111111110110001111111111111111110110001101101"
                        "1111010100"
                        "010010011101101010101010101010101111111111111111010010"
                        "0111111111"
                        "111111111111000110110111110101000100100111111111111111"
                        "1111100001"
                        "111111111111111111010001111111010110101010100010111111"
                        "1111110111"
                        "010010011111111111111111110100011011111101010100010010"
                        "0111111111"
                        "111101111101000111111111111111101100100101011011100111"
                        "0001101010"
                        "111111111111011111010001111010010001110000101010111011"
                        "0100111100"
                        "011010101111111111110110110100011101000100010100010010"
                        "0111111111"
                        "111101101100000111111111111101101100000101011011001111"
                        "1010101001"
                        "111010010011011001001010111011010011011011001010101011"
                        "0100110110"
                        "11000001010010010011011011000010") {
                        std::cout << "Missing Child: " << subtask->second.capture_set().to_string() << std::endl;
                    }

                    std::cout << "Left Bounds: [" << subtask->second.lowerbound() << ", "
                              << subtask->second.upperbound() << "], Left Scope: [" << subtask->second.lowerscope()
                              << ", " << subtask->second.upperscope() << "]" << std::endl;
                    uncertainty = subtask->second.uncertainty();
                    subtask.release();
                }

                if (uncertainty > 0.0 && diagnose_non_convergence(key->second)) {
                    break;
                }
            }
            if (found == false) {
                std::cout << "Left Child Not Found." << std::endl;
            }
        }
        {
            vertex_accessor child;
            child_accessor key;
            bool found = false;
            if (m_graph.children.find(key, std::make_pair(task->second.capture_set(), (feature + 1)))) {
                float uncertainty = 0.0;
                vertex_accessor subtask;
                if (m_graph.vertices.find(subtask, key->second)) {
                    found = true;
                    std::cout << "Right Bounds: [" << subtask->second.lowerbound() << ", "
                              << subtask->second.upperbound() << "], Right Scope: [" << subtask->second.lowerscope()
                              << ", " << subtask->second.upperscope() << "]" << std::endl;
                    uncertainty = subtask->second.uncertainty();
                    subtask.release();
                }
                if (uncertainty > 0.0 && diagnose_non_convergence(key->second)) {
                    break;
                }
            }
            if (found == false) {
                std::cout << "Right Child Not Found." << std::endl;
            }
        }
    }

    if (reasons == 0) {
        std::cout << "Missing an upward call:" << std::endl;
        std::cout << task->second.inspect() << std::endl;
    }
    return true;
}
